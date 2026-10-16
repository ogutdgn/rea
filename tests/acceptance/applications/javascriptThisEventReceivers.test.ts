import { Client } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";
import { writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { expect } from "vitest";

import { toolContract } from "../../../src/contracts/toolContracts.js";
import {
  javascriptApplicationAnalysisResultSchema,
  type JavaScriptApplicationAnalysisResult,
} from "../../../src/domain/javascript/javascriptApplicationAnalysis.js";
import { createTestTempDirectory } from "../../fixtures/temporaryDirectory.js";
import { cliTest } from "../../support/cli/cliFixture.js";

// Each file is independent, so one unresolved class cannot hide another case.
const uncertainCases = [
  ["escaped", "", "unknownUse(a);"],
  ["replaced", "", "a.send = replacement;"],
  ["extracted", "", "const f = a.send; f.call(other);"],
  ["alias", "", "const alias = a; alias.send();"],
  ["exported-instance", "", "export { a };"],
  ["exported-class", "", "export { Bus };"],
  ["escaped-peer", "", "const peer = new Bus(); unknownUse(peer);"],
  ["constructor", "constructor() { return shared; }", ""],
  ["field", "field = unknown(this);", ""],
  ["static", "static register() { this.on('EVENT', () => {}); }", ""],
  ["prototype", "", "Bus.prototype.send = replacement;"],
  ["class-replaced", "", "Bus = replacement;"],
  ["eval", "", "eval(source);"],
  ["parser-recovery", "", "let duplicate; let duplicate;"],
  ["conditional-call", "", "if (flag) a.send();"],
  ["nested-call", "", "function later() { a.send(); }"],
  ["computed-call", "", "a['send']();"],
  [
    "inherited",
    "",
    "class Child extends Bus {} const child = new Child(); child.send();",
  ],
  [
    "repeated-allocation",
    "",
    "function create() { return new Bus(); } create(); create();",
  ],
  [
    "opaque-listener",
    "opaque() { this.on('EVENT', externalHandler); }",
    "a.opaque();",
  ],
] as const;

const receiverFixture = `import { EventEmitter } from "node:events";
class Producer extends EventEmitter {
  run() { this.on("ready", () => {}); }
}
class Unrelated extends EventEmitter {
  run() { this.emit("ready"); }
}
new Producer().run();
new Unrelated().run();
const stable = new EventEmitter();
stable.on("ready", () => {});
stable.emit("ready");
class Shared extends EventEmitter {
  register() { this.on("instances", handler); }
  dispatch() { this.emit("instances"); }
  static register() { this.on("static", handler); }
  arrow = () => this.on("arrows", handler);
  nested() {
    const lexical = () => this.emit("arrows");
    function dynamic() { this.emit("arrows"); }
  }
}
new Shared().register();
new Shared().dispatch();
class Derived extends Shared {
  static dispatch() { this.emit("static"); }
}
function rebound() { this.on("rebound", handler); }
rebound.call(stable);
rebound.apply(new EventEmitter());
rebound.bind(new EventEmitter())();
const object = {
  register() { this.on("object", handler); },
  dispatch() { this.emit("object"); },
};
class Base extends EventEmitter {
  register() { this.on("inheritance", handler); }
}
class Conditional extends (flag ? Base : Unrelated) {
  dispatch() { this.emit("inheritance"); }
}
this.emit(dynamicName);
class Known extends EventEmitter {
  listen() { this.on("connected", () => {}); }
  send() { this.emit("connected"); }
  run() { this.on("within-call", () => {}); this.emit("within-call"); }
}
const a = new Known();
const b = new Known();
a.listen();
a.send();
a.send(); // Repeated calls must not duplicate the same receiver observation.
b.send();
a.run();
class Other extends EventEmitter { send() { this.emit("connected"); } }
const c = new Other();
c.send();
class Uncertain extends EventEmitter {
  listen() { this.on("uncertain", () => {}); }
  send() { this.emit("uncertain"); }
}
const escaped = new Uncertain();
unknownUse(escaped);
escaped.listen();
escaped.send();
const replaced = new Uncertain();
replaced.send = other;
replaced.listen();
replaced.send();
function unknownReceiver(value) { value.send(); }

`;

type Graph = JavaScriptApplicationAnalysisResult["semantic_graph"];

const assertOriginalReceivers = (graph: Graph): void => {
  const events = graph.nodes.filter(({ kind }) => kind === "event");
  const stable = events.filter(
    ({ properties }) =>
      String(properties.emitter_key).startsWith("binding:") &&
      properties.event_name === "ready",
  );
  expect(stable).toHaveLength(1);
  const stableEvent = stable[0]!;
  expect(stableEvent.properties.event_name).toBe("ready");
  const stableRelations = graph.relations.filter(
    (relation) =>
      relation.source_node_id === stableEvent.node_id ||
      relation.target_node_id === stableEvent.node_id,
  );
  expect(stableRelations.map(({ relation }) => relation).sort()).toEqual([
    "dispatches-candidate",
    "registers-listener",
  ]);
  expect(
    stableRelations.every(({ resolution }) => resolution === "resolved"),
  ).toBe(true);
  expect(
    graph.unknowns.some(({ node_id }) => node_id === stableEvent.node_id),
  ).toBe(false);
  const originalNames = new Set([
    "ready",
    "instances",
    "static",
    "arrows",
    "rebound",
    "object",
    "inheritance",
    null,
  ]);
  const receivers = events.filter(
    ({ node_id, properties }) =>
      node_id !== stableEvent.node_id &&
      originalNames.has(properties.event_name as string | null),
  );
  // These are source occurrences, not a claim that runtime objects differ.
  expect(receivers).toHaveLength(15);
  expect(
    receivers.filter(({ properties }) => properties.event_name === "ready"),
  ).toHaveLength(2);
  for (const event of receivers) {
    const relations = graph.relations.filter(
      (relation) =>
        relation.source_node_id === event.node_id ||
        relation.target_node_id === event.node_id,
    );
    expect(relations).toHaveLength(1);
    expect(relations[0]?.resolution).toBe("candidate");
    expect(graph.unknowns).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          node_id: event.node_id,
          family: "event",
          reason: "ambiguous-target",
          relation_kinds: [relations[0]?.relation],
        }),
      ]),
    );
  }
  expect(
    receivers.filter(({ properties }) => properties.event_name === null),
  ).toHaveLength(1);
};

const assertResolvedReceivers = (graph: Graph): void => {
  const events = graph.nodes.filter(({ kind }) => kind === "event");
  const connected = events.filter(
    ({ properties }) =>
      properties.event_name === "connected" &&
      String(properties.emitter_key).startsWith("binding:"),
  );
  expect(connected).toHaveLength(3);
  for (const [name, kinds] of [
    ["a", ["dispatches-candidate", "registers-listener"]],
    ["b", ["dispatches-candidate"]],
    ["c", ["dispatches-candidate"]],
  ] as const) {
    const event = connected.find(({ properties }) =>
      String(properties.emitter_key).endsWith(`:binding:${name}`),
    );
    expect(event).toBeDefined();
    const relations = graph.relations.filter(
      (relation) =>
        relation.source_node_id === event!.node_id ||
        relation.target_node_id === event!.node_id,
    );
    expect(relations.map(({ relation }) => relation).sort()).toEqual(kinds);
    expect(relations.every(({ resolution }) => resolution === "resolved")).toBe(
      true,
    );
    expect(
      graph.unknowns.some(({ node_id }) => node_id === event!.node_id),
    ).toBe(false);
    for (const relation of relations) {
      expect(relation.evidence.location).toMatchObject({
        available: true,
        value: { kind: "source-range" },
      });
      expect(relation.properties.receiver_binding_id).toEqual(
        String(event!.properties.emitter_key).slice("binding:".length),
      );
      expect(relation.properties.event_source_range).toHaveProperty(
        "start.line",
      );
    }
  }
};

const assertUncertainReceivers = (graph: Graph): void => {
  const events = graph.nodes.filter(({ kind }) => kind === "event");
  // Generic method observations stay uncertain: calls from outside the
  // supported local subset must not inherit a's or b's identity.
  const generic = events.filter(
    ({ properties }) =>
      properties.event_name === "connected" &&
      String(properties.emitter_key).startsWith("expression:"),
  );
  expect(generic).toHaveLength(3);
  for (const event of generic) {
    expect(
      graph.unknowns.some(
        ({ node_id, family }) =>
          node_id === event.node_id && family === "event",
      ),
    ).toBe(true);
  }
  const uncertain = events.filter(
    ({ properties }) => properties.event_name === "uncertain",
  );
  expect(uncertain).toHaveLength(2);
  expect(
    uncertain.every(({ properties }) =>
      String(properties.emitter_key).startsWith("expression:"),
    ),
  ).toBe(true);
  for (const event of uncertain) {
    const relations = graph.relations.filter(
      (relation) =>
        relation.source_node_id === event.node_id ||
        relation.target_node_id === event.node_id,
    );
    expect(
      relations.every(({ resolution }) => resolution === "candidate"),
    ).toBe(true);
    expect(
      graph.unknowns.some(
        ({ node_id, reason }) =>
          node_id === event.node_id && reason === "ambiguous-target",
      ),
    ).toBe(true);
  }
  const within = events.filter(
    ({ properties }) =>
      properties.event_name === "within-call" &&
      String(properties.emitter_key).startsWith("binding:"),
  );
  expect(within).toHaveLength(1);
  expect(
    graph.relations
      .filter(
        (relation) =>
          relation.source_node_id === within[0]!.node_id ||
          relation.target_node_id === within[0]!.node_id,
      )
      .map(({ relation, resolution }) => [relation, resolution])
      .sort(),
  ).toEqual([
    ["dispatches-candidate", "resolved"],
    ["registers-listener", "resolved"],
  ]);
  for (const [name] of uncertainCases) {
    const candidates = events.filter(
      ({ properties }) => properties.event_name === `uncertain-${name}`,
    );
    expect(candidates.length, name).toBeGreaterThanOrEqual(2);
    for (const event of candidates) {
      expect(String(event.properties.emitter_key), name).toMatch(
        /^expression:/u,
      );
      const relations = graph.relations.filter(
        (relation) =>
          relation.source_node_id === event.node_id ||
          relation.target_node_id === event.node_id,
      );
      expect(
        relations.every(({ resolution }) => resolution === "candidate"),
        name,
      ).toBe(true);
      expect(
        graph.unknowns.some(
          ({ node_id, reason }) =>
            node_id === event.node_id && reason === "ambiguous-target",
        ),
        name,
      ).toBe(true);
    }
  }
};

cliTest(
  "resolves proven local this receivers and preserves uncertain receivers through CLI and stdio MCP",
  async ({ cli }) => {
    const root = await createTestTempDirectory("rea-this-event-receivers-");
    // Inert input: even a single method can run with different receivers.
    await writeFile(join(root, "main.mjs"), receiverFixture);
    for (const [name, extraMember, extraUse] of uncertainCases) {
      const source = `import { EventEmitter } from "node:events";
class Bus extends EventEmitter {
  listen() { this.on('EVENT', () => {}); }
  send() { this.emit('EVENT'); }
  ${extraMember}
}
const a = new Bus();
${extraUse}
a.listen();
a.send();`;
      await writeFile(
        join(root, `${name}.mjs`),
        source.replaceAll("EVENT", `uncertain-${name}`),
      );
    }
    const cliResponse = await cli.run({
      arguments: ["analyze-javascript-application", root, "--json"],
      timeoutMs: 60_000,
    });
    expect(cliResponse.exitCode).toBe(0);
    const contract = toolContract("analyze_javascript_application");
    const cliEvidence = contract.outputSchema.parse(cliResponse.json);
    const client = new Client({ name: "this-event-receivers", version: "1" });
    const transport = new StdioClientTransport({
      command: process.execPath,
      args: [resolve("scripts/rea.mjs"), "mcp"],
      cwd: process.cwd(),
      env: { PATH: process.env.PATH ?? "" },
      stderr: "pipe",
    });
    transport.stderr?.on("data", () => undefined);
    try {
      await client.connect(transport);
      const response = await client.callTool({
        name: "analyze_javascript_application",
        arguments: { input_path: root },
      });
      expect(response.isError).not.toBe(true);
      const mcpEvidence = contract.outputSchema.parse(
        response.structuredContent,
      );
      const text = response.content.find((item) => item.type === "text");
      if (text?.type !== "text") throw new Error("Missing MCP text result");
      expect(contract.outputSchema.parse(JSON.parse(text.text))).toEqual(
        mcpEvidence,
      );
      expect(mcpEvidence.evidence_id).toBe(cliEvidence.evidence_id);
      const graph = javascriptApplicationAnalysisResultSchema.parse(
        mcpEvidence.normalized_result,
      ).semantic_graph;
      expect(graph).toEqual(
        javascriptApplicationAnalysisResultSchema.parse(
          cliEvidence.normalized_result,
        ).semantic_graph,
      );
      assertOriginalReceivers(graph);
      assertResolvedReceivers(graph);
      assertUncertainReceivers(graph);
      await client.ping();
    } finally {
      await client.close();
      await transport.close();
    }
  },
  120_000,
);

cliTest(
  "reports bounded receiver expansion without dropping uncertain source observations",
  async ({ cli }) => {
    const root = await createTestTempDirectory("rea-this-receiver-bound-");
    await writeFile(
      join(root, "main.mjs"),
      `
import { EventEmitter } from "node:events";
class Bus extends EventEmitter {
  run() { this.on("bounded", () => {}); this.emit("bounded"); }
}
${Array.from({ length: 257 }, (_, i) => `const b${i} = new Bus(); b${i}.run();`).join("\n")}
`,
    );
    const response = await cli.run({
      arguments: ["analyze-javascript-application", root, "--json"],
      timeoutMs: 60_000,
    });
    expect(response.exitCode).toBe(0);
    const evidence = toolContract(
      "analyze_javascript_application",
    ).outputSchema.parse(response.json);
    const graph = javascriptApplicationAnalysisResultSchema.parse(
      evidence.normalized_result,
    ).semantic_graph;
    const events = graph.nodes.filter(({ kind }) => kind === "event");
    expect(
      events.filter(({ properties }) =>
        String(properties.emitter_key).startsWith("binding:"),
      ),
    ).toHaveLength(256);
    expect(
      events.filter(({ properties }) =>
        String(properties.emitter_key).startsWith("expression:"),
      ),
    ).toHaveLength(2);
    expect(graph.unknowns).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ family: "event", reason: "resource-limit" }),
        expect.objectContaining({
          family: "event",
          reason: "ambiguous-target",
        }),
      ]),
    );
    expect(JSON.stringify(graph)).toContain(
      "javascript_semantic_event_receiver_expansion",
    );
  },
  120_000,
);
