import * as t from "@babel/types";

import { builtinDataEffectMethod } from "./javascriptSemanticDataEffectHelpers.js";
import type { JavaScriptSemanticEventOperation } from "./javascriptSemanticIr.js";
import { SEMANTIC_EVENT_RECEIVER_EXPANSION_LIMIT } from "./javascriptSemanticResourceLimits.js";
import {
  resolveSemanticBindingState,
  semanticStableInitializer,
  type JavaScriptSemanticAnalysisState,
  type JavaScriptSemanticBindingState,
} from "./javascriptSemanticState.js";
import { traverseJavaScriptAst } from "./javascriptSemanticTraversal.js";
import { range } from "./javascriptStaticAnalysisHelpers.js";

type Event = JavaScriptSemanticEventOperation;
type ClassMethods = ReadonlyMap<string, readonly Event[]>;

/**
 * Specialize event observations for closed, module-local instance calls.
 * Source-level observations remain partial: a method is not a runtime object.
 * No target code is executed and no class/function key claims receiver equality.
 */
export const resolveSemanticEventReceivers = (
  state: JavaScriptSemanticAnalysisState,
  events: readonly Event[],
  eventSites: ReadonlyMap<t.CallExpression | t.OptionalCallExpression, Event>,
  eventMethods: readonly string[],
): readonly Event[] => {
  if (eventSites.size === 0) return events;
  const references = new Map<string, t.Identifier[]>();
  let dynamicScope = false;
  traverseJavaScriptAst(state.program, {
    enter(node, parent) {
      // Direct eval can replace otherwise closed local methods invisibly.
      if (
        t.isWithStatement(node) ||
        (t.isCallExpression(node) &&
          t.isIdentifier(node.callee, { name: "eval" }))
      )
        dynamicScope = true;
      if (
        !t.isIdentifier(node) ||
        parent === null ||
        !t.isReferenced(node, parent)
      )
        return;
      const binding = resolveSemanticBindingState(state, node, node.name);
      if (binding === undefined) return;
      const uses = references.get(binding.bindingId) ?? [];
      uses.push(node);
      references.set(binding.bindingId, uses);
    },
  });
  if (dynamicScope) return events;
  const classes = new Map<string, ClassMethods | null>();
  const output = [...events];
  for (const declaration of state.program.body) {
    if (!t.isVariableDeclaration(declaration, { kind: "const" })) continue;
    for (const item of declaration.declarations) {
      if (
        !t.isIdentifier(item.id) ||
        !t.isNewExpression(item.init) ||
        item.init.arguments.length !== 0 ||
        !t.isIdentifier(item.init.callee)
      )
        continue;
      const instance = resolveSemanticBindingState(
        state,
        item.id,
        item.id.name,
      );
      const owner = resolveSemanticBindingState(
        state,
        item.init.callee,
        item.init.callee.name,
      );
      if (
        instance === undefined ||
        owner === undefined ||
        !stable(instance, state) ||
        !stable(owner, state)
      )
        continue;
      let methods = classes.get(owner.bindingId);
      if (methods === undefined) {
        methods = classMethods(
          owner,
          state,
          references,
          eventSites,
          eventMethods,
        );
        classes.set(owner.bindingId, methods);
      }
      if (methods === null) continue;
      const calls = instanceCalls(instance, methods, references, state, item);
      if (calls === null) continue;
      for (const [name, call] of calls) {
        for (const event of methods.get(name) ?? []) {
          if (
            output.length - events.length >=
            SEMANTIC_EVENT_RECEIVER_EXPANSION_LIMIT
          ) {
            state.valueResourceLimits.add("event-receiver-expansion");
            return output;
          }
          output.push({
            ...event,
            eventId: `${event.eventId}:receiver:${instance.bindingId}`,
            emitterKey: `binding:${instance.bindingId}`,
            emitterBindingId: instance.bindingId,
            receiverCallLocation: range(call),
            resolution: event.eventName === null ? "unresolved" : "complete",
          });
        }
      }
    }
  }
  return output;
};

const stable = (
  binding: JavaScriptSemanticBindingState,
  state: JavaScriptSemanticAnalysisState,
): boolean =>
  !binding.mutable &&
  (state.bindingWrites.get(binding.bindingId)?.length ?? 0) === 0 &&
  semanticStableInitializer(binding, state) !== undefined;

const classMethods = (
  binding: JavaScriptSemanticBindingState,
  state: JavaScriptSemanticAnalysisState,
  references: ReadonlyMap<string, readonly t.Identifier[]>,
  eventSites: ReadonlyMap<t.CallExpression | t.OptionalCallExpression, Event>,
  eventMethods: readonly string[],
): ClassMethods | null => {
  const klass = semanticStableInitializer(binding, state)?.node;
  // A default EventEmitter constructor cannot substitute an arbitrary object.
  // Custom constructors, inherited methods, fields and decorators need their
  // own receiver/side-effect proofs and deliberately stay uncertain here.
  if (
    !t.isClassDeclaration(klass) ||
    state.parentsByNode.get(klass) !== state.program ||
    (klass.decorators?.length ?? 0) > 0 ||
    klass.superClass === null ||
    klass.superClass === undefined ||
    builtinDataEffectMethod(klass.superClass, state, ["node:events"]) !==
      "EventEmitter"
  )
    return null;
  if (
    !(references.get(binding.bindingId) ?? []).every((use) => {
      const parent = state.parentsByNode.get(use);
      return t.isNewExpression(parent) && parent.callee === use;
    })
  )
    return null;
  const methods = new Map<string, readonly Event[]>();
  for (const member of klass.body.body) {
    if (
      !t.isClassMethod(member) ||
      member.kind !== "method" ||
      member.static ||
      member.computed ||
      member.async ||
      member.generator ||
      member.params.length !== 0 ||
      (member.decorators?.length ?? 0) > 0 ||
      !t.isIdentifier(member.key) ||
      methods.has(member.key.name) ||
      eventMethods.includes(member.key.name)
    )
      return null;
    const operations: Event[] = [];
    for (const statement of member.body.body) {
      if (
        !t.isExpressionStatement(statement) ||
        !t.isCallExpression(statement.expression)
      )
        return null;
      const call = statement.expression;
      const event = eventSites.get(call);
      if (
        event === undefined ||
        !t.isMemberExpression(call.callee) ||
        call.callee.computed ||
        !t.isThisExpression(call.callee.object) ||
        !call.arguments.every(inertEventArgument)
      )
        return null;
      operations.push(event);
    }
    methods.set(member.key.name, operations);
  }
  // An escaped instance exposes the shared prototype too. Do not resolve
  // other instances of this class using a method table that may be replaced.
  for (const use of references.get(binding.bindingId) ?? []) {
    const allocation = state.parentsByNode.get(use);
    const item =
      allocation === undefined
        ? undefined
        : state.parentsByNode.get(allocation);
    const declaration =
      item === undefined ? undefined : state.parentsByNode.get(item);
    if (
      !t.isNewExpression(allocation) ||
      allocation.arguments.length !== 0 ||
      !t.isVariableDeclarator(item) ||
      item.init !== allocation ||
      !t.isIdentifier(item.id) ||
      !t.isVariableDeclaration(declaration, { kind: "const" }) ||
      state.parentsByNode.get(declaration) !== state.program
    )
      return null;
    const instance = resolveSemanticBindingState(state, item.id, item.id.name);
    if (
      instance === undefined ||
      !stable(instance, state) ||
      instanceCalls(instance, methods, references, state, item) === null
    )
      return null;
  }
  return methods;
};

/** An opaque callback can receive and mutate the emitter during dispatch. */
const inertEventArgument = (node: t.Node): boolean =>
  t.isStringLiteral(node) ||
  t.isNumericLiteral(node) ||
  t.isBooleanLiteral(node) ||
  t.isNullLiteral(node) ||
  t.isBigIntLiteral(node) ||
  ((t.isArrowFunctionExpression(node) || t.isFunctionExpression(node)) &&
    node.params.length === 0 &&
    t.isBlockStatement(node.body) &&
    node.body.body.length === 0);

const instanceCalls = (
  binding: JavaScriptSemanticBindingState,
  methods: ClassMethods,
  references: ReadonlyMap<string, readonly t.Identifier[]>,
  state: JavaScriptSemanticAnalysisState,
  declaration: t.VariableDeclarator,
): ReadonlyMap<string, t.CallExpression> | null => {
  const calls = new Map<string, t.CallExpression>();
  for (const use of references.get(binding.bindingId) ?? []) {
    const member = state.parentsByNode.get(use);
    const call =
      member === undefined ? undefined : state.parentsByNode.get(member);
    const statement =
      call === undefined ? undefined : state.parentsByNode.get(call);
    // Reading/aliasing/passing the receiver or replacing/extracting a method
    // breaks closure. Nested calls can run after unknown mutation or repeatedly.
    if (
      !t.isMemberExpression(member) ||
      member.object !== use ||
      member.computed ||
      !t.isIdentifier(member.property) ||
      !methods.has(member.property.name) ||
      !t.isCallExpression(call) ||
      call.callee !== member ||
      call.arguments.length !== 0 ||
      !t.isExpressionStatement(statement) ||
      state.parentsByNode.get(statement) !== state.program ||
      (call.start ?? -1) <= (declaration.end ?? -1)
    )
      return null;
    if (!calls.has(member.property.name)) calls.set(member.property.name, call);
  }
  return calls;
};
