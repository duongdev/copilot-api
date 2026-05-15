import type { Context } from "hono"

import consola from "consola"
import { streamSSE } from "hono/streaming"

import { awaitApproval } from "~/lib/approval"
import { checkRateLimit } from "~/lib/rate-limit"
import { state } from "~/lib/state"
import {
  createChatCompletions,
  type ChatCompletionChunk,
  type ChatCompletionResponse,
} from "~/services/copilot/create-chat-completions"

import {
  type AnthropicMessagesPayload,
  type AnthropicStreamEventData,
  type AnthropicStreamState,
} from "./anthropic-types"
import {
  translateToAnthropic,
  translateToOpenAI,
} from "./non-stream-translation"
import { translateChunkToAnthropicEvents } from "./stream-translation"

export async function handleCompletion(c: Context) {
  await checkRateLimit(state)

  const anthropicPayload = await c.req.json<AnthropicMessagesPayload>()
  consola.debug("Anthropic request payload:", JSON.stringify(anthropicPayload))

  const openAIPayload = translateToOpenAI(anthropicPayload)
  consola.debug(
    "Translated OpenAI request payload:",
    JSON.stringify(openAIPayload),
  )

  if (state.manualApprove) {
    await awaitApproval()
  }

  const response = await createChatCompletions(openAIPayload)

  if (isNonStreaming(response)) {
    consola.debug(
      "Non-streaming response from Copilot:",
      JSON.stringify(response).slice(-400),
    )
    const anthropicResponse = translateToAnthropic(response)
    consola.debug(
      "Translated Anthropic response:",
      JSON.stringify(anthropicResponse),
    )
    return c.json(anthropicResponse)
  }

  consola.debug("Streaming response from Copilot")
  return streamSSE(c, async (stream) => {
    const streamState: AnthropicStreamState = {
      messageStartSent: false,
      contentBlockIndex: 0,
      contentBlockOpen: false,
      toolCalls: {},
    }

    // Buffer all events so we can backfill usage from the final chunk.
    // OpenAI streams usage in a trailing chunk with empty choices — separate from
    // the finish_reason chunk — so both message_start and message_delta need patching.
    const buffered: Array<AnthropicStreamEventData> = []
    let finalUsage: ChatCompletionChunk["usage"] | undefined

    for await (const rawEvent of response) {
      consola.debug("Copilot raw stream event:", JSON.stringify(rawEvent))
      if (rawEvent.data === "[DONE]") {
        break
      }

      if (!rawEvent.data) {
        continue
      }

      let chunk: ChatCompletionChunk
      try {
        chunk = JSON.parse(rawEvent.data) as ChatCompletionChunk
      } catch {
        consola.warn("Failed to parse stream chunk, skipping:", rawEvent.data)
        continue
      }

      if (chunk.usage) {
        finalUsage = chunk.usage
      }
      buffered.push(...translateChunkToAnthropicEvents(chunk, streamState))
    }

    if (finalUsage) {
      const cached = finalUsage.prompt_tokens_details?.cached_tokens ?? 0
      const usagePatch = {
        input_tokens: finalUsage.prompt_tokens - cached,
        output_tokens: finalUsage.completion_tokens,
        ...(cached > 0 && { cache_read_input_tokens: cached }),
      }

      for (const event of buffered) {
        if (event.type === "message_start") {
          event.message.usage = {
            ...usagePatch,
            output_tokens: 0, // always 0 in message_start per Anthropic spec
          }
        } else if (event.type === "message_delta") {
          event.usage = usagePatch
        }
      }
    }

    for (const event of buffered) {
      consola.debug("Translated Anthropic event:", JSON.stringify(event))
      await stream.writeSSE({
        event: event.type,
        data: JSON.stringify(event),
      })
    }
  })
}

const isNonStreaming = (
  response: Awaited<ReturnType<typeof createChatCompletions>>,
): response is ChatCompletionResponse => Object.hasOwn(response, "choices")
