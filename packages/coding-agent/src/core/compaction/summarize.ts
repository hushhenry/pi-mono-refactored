import { generateText, type LanguageModel, type CoreMessage } from "ai";
import { type AgentMessage } from "../../../../../ai/src/types.js";
import { 
    createFileOps, 
    extractFileOpsFromMessage, 
    computeFileLists, 
    formatFileOperations, 
    serializeConversation, 
    SUMMARIZATION_SYSTEM_PROMPT,
    estimateTokens
} from "./utils.js";

const SUMMARIZATION_PROMPT = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

export interface SummarizeOptions {
	model: LanguageModel;
	keepRecentTokens?: number;
    reserveTokens?: number;
    previousSummary?: string;
}

export async function summarizeHistory(
    messages: AgentMessage[], 
    options: SummarizeOptions
): Promise<{ messages: AgentMessage[]; summary: string }> {
	if (!options.model) throw new Error("SummarizeStrategy requires a model");

	const keepRecentTokens = options.keepRecentTokens ?? 20000;
	
    // Find cut point based on tokens
    let accumulatedTokens = 0;
    let cutIndex = 0;
    for (let i = messages.length - 1; i >= 0; i--) {
        accumulatedTokens += estimateTokens(messages[i]);
        if (accumulatedTokens >= keepRecentTokens) {
            cutIndex = i;
            break;
        }
    }

	const toSummarize = messages.slice(0, cutIndex);
	const toKeep = messages.slice(cutIndex);

	if (toSummarize.length === 0) {
		return { messages, summary: options.previousSummary || "" };
	}

	const fileOps = createFileOps();
	for (const msg of toSummarize) {
		extractFileOpsFromMessage(msg, fileOps);
	}

	const llmHistory = toSummarize.map(({ timestamp, usage, ...rest }: any) => rest as CoreMessage);
	const conversationText = serializeConversation(llmHistory);

    let promptText = `<conversation>\n${conversationText}\n</conversation>\n\n`;
    if (options.previousSummary) {
        promptText += `<previous-summary>\n${options.previousSummary}\n</previous-summary>\n\n`;
    }
    promptText += SUMMARIZATION_PROMPT;

	const { text } = await generateText({
		model: options.model,
		system: SUMMARIZATION_SYSTEM_PROMPT,
		messages: [
			{ role: "user", content: promptText }
		],
	});

	const { readFiles, modifiedFiles } = computeFileLists(fileOps);
	const summaryWithFiles = text + formatFileOperations(readFiles, modifiedFiles);

	const summaryMessage: AgentMessage = {
		role: "user",
		content: `[Context Summary of prior turns]:\n${summaryWithFiles}`,
		timestamp: Date.now()
	};

	return {
		messages: [summaryMessage, ...toKeep],
		summary: text
	};
}
