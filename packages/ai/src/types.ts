import {
	type LanguageModel,
	type CoreMessage,
	type CoreUserMessage,
	type CoreAssistantMessage,
	type CoreToolMessage,
} from "ai";
import type { TSchema } from "@sinclair/typebox";
import { AssistantMessageEventStream } from "./utils/event-stream.js";

export type { AssistantMessageEventStream } from "./utils/event-stream.js";

export type Api = string;
export type Provider = string;

export type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh";

/** Token budgets for each thinking level (token-based providers only) */
export interface ThinkingBudgets {
	minimal?: number;
	low?: number;
	medium?: number;
	high?: number;
}

export interface StreamOptions {
	temperature?: number;
	maxTokens?: number;
	signal?: AbortSignal;
	apiKey?: string;
	headers?: Record<string, string>;
	metadata?: Record<string, unknown>;
    /**
	 * Prompt cache retention preference. Providers map this to their supported values.
	 * Default: "short".
	 */
	cacheRetention?: "none" | "short" | "long";
}

export interface SimpleStreamOptions extends StreamOptions {
	reasoning?: ThinkingLevel;
	thinkingBudgets?: ThinkingBudgets;
}

/**
 * Metadata added to standard Vercel AI SDK messages for session tracking.
 */
export interface MessageMeta {
	timestamp: number;
	usage?: Usage;
}

export interface Usage {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	totalTokens: number;
	cost: {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
		total: number;
	};
}

export type StopReason = "stop" | "length" | "tool-call" | "error" | "abort";

export type AgentUserMessage = CoreUserMessage & MessageMeta;
export type AgentAssistantMessage = CoreAssistantMessage & MessageMeta;
export type AgentToolMessage = CoreToolMessage & MessageMeta;

export type Message = AgentUserMessage | AgentAssistantMessage | AgentToolMessage;
export type AgentMessage = Message;

export interface Tool<TParameters extends TSchema = TSchema> {
	name: string;
	description: string;
	parameters: TParameters;
}

export interface Context {
	systemPrompt?: string;
	messages: Message[];
	tools?: Tool[];
}

export interface Model<TApi extends Api = Api> {
	id: string;
	name: string;
	api: TApi;
	provider: Provider;
	baseUrl: string;
	reasoning: boolean;
	input: ("text" | "image")[];
	cost: {
		input: number;
		output: number;
		cacheRead: number; // $/million tokens
		cacheWrite: number; // $/million tokens
	};
	contextWindow: number;
	maxTokens: number;
}

export type AssistantMessageEvent =
	| { type: "start"; partial: AgentAssistantMessage }
	| { type: "text_start"; contentIndex: number; partial: AgentAssistantMessage }
	| { type: "text_delta"; contentIndex: number; delta: string; partial: AgentAssistantMessage }
	| { type: "text_end"; contentIndex: number; content: string; partial: AgentAssistantMessage }
	| { type: "thinking_start"; contentIndex: number; partial: AgentAssistantMessage }
	| { type: "thinking_delta"; contentIndex: number; delta: string; partial: AgentAssistantMessage }
	| { type: "thinking_end"; contentIndex: number; content: string; partial: AgentAssistantMessage }
	| { type: "toolcall_start"; contentIndex: number; partial: AgentAssistantMessage }
	| { type: "toolcall_delta"; contentIndex: number; delta: string; partial: AgentAssistantMessage }
	| { type: "toolcall_end"; contentIndex: number; toolCall: any; partial: AgentAssistantMessage }
	| { type: "done"; reason: any; message: AgentAssistantMessage }
	| { type: "error"; reason: any; error: AgentAssistantMessage };
