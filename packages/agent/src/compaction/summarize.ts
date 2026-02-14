import { generateText, type LanguageModel, type CoreMessage } from "ai";
import { type AgentMessage } from "../types.js";
import { type CompactionResult, type CompactionStrategy } from "./types.js";
import { 
    createFileOps, 
    extractFileOpsFromMessage, 
    computeFileLists, 
    formatFileOperations, 
    serializeConversation, 
    SUMMARIZATION_SYSTEM_PROMPT 
} from "./utils.js";

const SUMMARIZATION_PROMPT = `The messages above are a conversation to summarize. Create a structured context checkpoint summary.
Use this EXACT format:
## Goal
[What is the user trying to accomplish?]
## Progress
### Done
- [x] [Completed tasks]
## Key Decisions
- **[Decision]**: [Rationale]
## Next Steps
1. [What should happen next]`;

export interface SummarizeOptions {
	model: LanguageModel;
	keepRecent?: number;
}

export class SummarizeStrategy implements CompactionStrategy {
	name = "summarize";

	async compact(messages: AgentMessage[], options: SummarizeOptions): Promise<CompactionResult> {
		if (!options.model) throw new Error("SummarizeStrategy requires a model");

		const keepRecent = options.keepRecent ?? 6;
		const boundary = Math.max(0, messages.length - keepRecent);
		const toSummarize = messages.slice(0, boundary);
		const toKeep = messages.slice(boundary);

		if (toSummarize.length === 0) {
			return { messages, compactedCount: 0, tokensBefore: 0, tokensAfter: 0 };
		}

		const fileOps = createFileOps();
		for (const msg of toSummarize) {
			extractFileOpsFromMessage(msg, fileOps);
		}

		const llmHistory = toSummarize.map(({ timestamp, usage, ...rest }: any) => rest as CoreMessage);
		const conversationText = serializeConversation(llmHistory);

		const { text } = await generateText({
			model: options.model,
			system: SUMMARIZATION_SYSTEM_PROMPT,
			messages: [
				{ role: "user", content: `<conversation>\n${conversationText}\n</conversation>\n\n${SUMMARIZATION_PROMPT}` }
			],
		});

		const { readFiles, modifiedFiles } = computeFileLists(fileOps);
		const summary = text + formatFileOperations(readFiles, modifiedFiles);

		const summaryMessage: AgentMessage = {
			role: "user",
			content: `[Context Summary of prior turns]:\n${summary}`,
			timestamp: Date.now()
		};

		return {
			messages: [summaryMessage, ...toKeep],
			compactedCount: toSummarize.length,
			tokensBefore: 0, 
			tokensAfter: 0
		};
	}
}
