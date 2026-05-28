import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  buildRouteArguments,
  buildAmapAgentMessages,
  buildRouteSummaryMessages,
  extractTextContent,
  parseArgs,
  pickRouteTool,
  toOpenAiToolDefinitions
} from "../demos/amap_route_demo.js";

describe("Amap route MCP demo helpers", () => {
  it("defaults to Beijing Xierqi to Shenzhen transit planning", () => {
    const options = parseArgs([]);

    assert.equal(options.mode, "transit");
    assert.equal(options.origin, "116.306295,40.053034");
    assert.equal(options.destination, "114.029113,22.609767");
    assert.equal(options.city, "北京");
    assert.equal(options.cityd, "深圳");
    assert.equal(options.raw, false);
    assert.equal(options.includeFlight, true);
    assert.equal(options.departureAirport, "北京首都国际机场");
    assert.equal(options.arrivalAirport, "深圳宝安国际机场");
    assert.equal(options.maxToolRounds, 5);
    assert.match(options.request, /西二旗/);
  });

  it("parses route demo CLI arguments", () => {
    const options = parseArgs([
      "--mode",
      "walking",
      "--origin",
      "121.1,31.1",
      "--destination",
      "121.2,31.2",
      "--city",
      "上海",
      "--request",
      "从上海人民广场去外滩",
      "--no-flight",
      "--raw"
    ]);

    assert.equal(options.mode, "walking");
    assert.equal(options.origin, "121.1,31.1");
    assert.equal(options.destination, "121.2,31.2");
    assert.equal(options.city, "上海");
    assert.equal(options.raw, true);
    assert.equal(options.includeFlight, false);
    assert.equal(options.request, "从上海人民广场去外滩");
  });

  it("picks the route tool for the requested mode", () => {
    const tool = pickRouteTool(
      [
        { name: "maps_geo" },
        { name: "maps_direction_driving" },
        { name: "maps_direction_walking" }
      ],
      "driving"
    );

    assert.equal(tool.name, "maps_direction_driving");
  });

  it("builds arguments from the discovered tool schema", () => {
    const args = buildRouteArguments(
      {
        name: "maps_direction_transit_integrated",
        inputSchema: {
          properties: {
            origin: {},
            destination: {},
            city: {},
            cityd: {}
          }
        }
      },
      {
        origin: "121.1,31.1",
        destination: "121.2,31.2",
        city: "上海",
        cityd: "杭州",
        mode: "transit",
        includeFlight: false,
        departureAirport: "上海虹桥机场",
        departureAirportCoord: "121.33426,31.19692",
        arrivalAirport: "杭州萧山机场",
        arrivalAirportCoord: "120.43333,30.23611",
        raw: false,
        request: "从上海去杭州",
        maxToolRounds: 4
      }
    );

    assert.deepEqual(args, {
      origin: "121.1,31.1",
      destination: "121.2,31.2",
      city: "上海",
      cityd: "杭州"
    });
  });

  it("extracts text content from MCP tool results", () => {
    const text = extractTextContent([
      { type: "text", text: "{\"route\":{\"distance\":\"2128602\"}}" }
    ]);

    assert.match(text, /2128602/);
  });

  it("builds a strict route summary prompt for the model", () => {
    const messages = buildRouteSummaryMessages(
      {
        origin: "116.306295,40.053034",
        destination: "114.029113,22.609767",
        city: "北京",
        cityd: "深圳",
        mode: "transit",
        includeFlight: true,
        departureAirport: "北京首都国际机场",
        departureAirportCoord: "116.615583,40.052657",
        arrivalAirport: "深圳宝安国际机场",
        arrivalAirportCoord: "113.814561,22.623291",
        raw: false,
        request: "从北京西二旗去深圳",
        maxToolRounds: 4
      },
      "maps_direction_transit_integrated",
      "{\"route\":{\"distance\":\"2128602\"}}"
    );

    assert.equal(messages.length, 2);
    assert.match(messages[0].content ?? "", /不能编造/);
    assert.match(messages[1].content ?? "", /推荐方案/);
    assert.match(messages[1].content ?? "", /maps_direction_transit_integrated/);
  });

  it("builds a model prompt that requires MCP tool use", () => {
    const messages = buildAmapAgentMessages(parseArgs([]));

    assert.equal(messages.length, 2);
    assert.match(messages[0].content ?? "", /必须通过工具/);
    assert.match(messages[1].content ?? "", /请先调用/);
    assert.match(messages[1].content ?? "", /飞机候选方案/);
    assert.match(messages[1].content ?? "", /北京首都国际机场/);
    assert.match(messages[1].content ?? "", /航班系统查询/);
  });

  it("converts discovered MCP tools to OpenAI-compatible tool definitions", () => {
    const tools = toOpenAiToolDefinitions([
      {
        name: "maps_direction_transit_integrated",
        description: "Transit route planning",
        inputSchema: {
          properties: {
            origin: {},
            destination: {}
          },
          required: ["origin", "destination"]
        }
      },
      { name: "unrelated_tool" }
    ]);

    assert.equal(tools.length, 1);
    assert.equal(tools[0].function.name, "maps_direction_transit_integrated");
    assert.deepEqual(tools[0].function.parameters?.required, ["origin", "destination"]);
  });
});
