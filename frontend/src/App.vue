<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Promotion } from '@element-plus/icons-vue'
import { formatMessage } from './formatMessage'

interface LawInfo {
  id: string
  name: string
  source_url: string
  article_count: number
}

interface Citation {
  law_name: string
  article_no: string
  hierarchy: string
  text: string
  source_url: string
}

interface HistoryMessage {
  role: 'user' | 'assistant'
  content: string
}

interface CitationIssue {
  law: string
  article_no: string
  reason: string
}

interface CitationVerifyInfo {
  passed: boolean
  citation_verified: boolean
  cited_count: number
  invalid_count: number
  precision: number
  hallucination: boolean
  action: 'pass' | 'rewrite_once' | 'fallback_chunks' | 'disabled'
  invalid: CitationIssue[]
  warnings: CitationIssue[]
}

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  streaming?: boolean
  isLegal?: boolean
  phase?: 'retrieving' | 'verifying'
  citationVerified?: boolean
  citationVerify?: CitationVerifyInfo
  answerRevised?: boolean
}

const REASON_LABELS: Record<string, string> = {
  not_in_kb: '不在知识库中',
  not_in_retrieved_chunks: '未在本次检索结果中',
}

const ACTION_LABELS: Record<string, string> = {
  pass: '引用校验通过',
  rewrite_once: '已自动修正法律依据',
  fallback_chunks: '已替换为检索到的法条',
  disabled: '校验未启用',
}

const DEFAULT_DISCLAIMER =
  '本系统由 AI 生成，仅供参考，不构成正式法律意见。具体案件请咨询执业律师。'

const SCROLL_THRESHOLD = 80

const input = ref('')
const loading = ref(false)
const laws = ref<LawInfo[]>([])
const messages = ref<ChatMessage[]>([])
const chatBodyRef = ref<HTMLElement | null>(null)
const disclaimer = ref(DEFAULT_DISCLAIMER)
const stickToBottom = ref(true)

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

function renderContent(msg: ChatMessage): string {
  if (msg.role === 'user') {
    return msg.content
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\n/g, '<br>')
  }
  return formatMessage(msg.content)
}

function typingHint(msg: ChatMessage): string {
  if (msg.isLegal === false) return '正在生成回答…'
  if (msg.phase === 'verifying') return '正在校验法律依据…'
  if (msg.phase === 'retrieving' || msg.content) return '正在生成回答…'
  return '正在识别问题并检索法条…'
}

function verifyTagType(msg: ChatMessage): 'success' | 'warning' | 'danger' | 'info' {
  const v = msg.citationVerify
  if (!v) {
    if (msg.citationVerified === false) return 'danger'
    if (msg.citationVerified) return 'success'
    return 'info'
  }
  if (!v.citation_verified || v.invalid_count > 0) return 'danger'
  if (v.action === 'rewrite_once' || v.action === 'fallback_chunks' || v.warnings.length) {
    return 'warning'
  }
  return 'success'
}

function verifyTagLabel(msg: ChatMessage): string {
  const v = msg.citationVerify
  if (!v) {
    if (msg.citationVerified === false) return '引用未通过校验'
    if (msg.citationVerified) return '引用已校验'
    return '引用校验'
  }
  if (!v.citation_verified) return '存在无法核实的引用'
  if (v.action !== 'pass') return ACTION_LABELS[v.action] ?? '引用已更新'
  if (v.warnings.length) return '引用已校验（含提示）'
  return '引用已校验'
}

function verifySummary(msg: ChatMessage): string {
  const v = msg.citationVerify
  if (!v) return ''
  const parts: string[] = []
  if (v.cited_count > 0) {
    parts.push(`共 ${v.cited_count} 条引用`)
  }
  if (v.action === 'pass' && !v.warnings.length) {
    parts.push('均在知识库中核实')
  }
  if (msg.answerRevised) {
    parts.push('回答已根据校验结果更新')
  }
  return parts.join(' · ')
}

function formatCitationIssue(item: CitationIssue): string {
  const reason = REASON_LABELS[item.reason] ?? item.reason
  return `《${item.law}》${item.article_no}（${reason}）`
}

function onChatScroll() {
  const el = chatBodyRef.value
  if (!el) return
  stickToBottom.value =
    el.scrollHeight - el.scrollTop - el.clientHeight < SCROLL_THRESHOLD
}

async function scrollToBottom(force = false) {
  if (!force && !stickToBottom.value) return
  await nextTick()
  const el = chatBodyRef.value
  if (el) el.scrollTop = el.scrollHeight
}

async function loadLaws() {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), 8000)
  try {
    const res = await fetch('/api/laws', { signal: controller.signal })
    if (res.ok) laws.value = await res.json()
  } catch {
    // 后端预热中或暂不可达时不阻塞页面
  } finally {
    window.clearTimeout(timer)
  }
}

function parseSseBlock(block: string): { event: string; data: string } | null {
  const lines = block.trim().split('\n')
  if (!lines.length) return null
  let event = 'message'
  let data = ''
  for (const line of lines) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    if (line.startsWith('data:')) data += line.slice(5).trim()
  }
  return data ? { event, data } : null
}

function buildHistory(): HistoryMessage[] {
  return messages.value
    .filter((m) => !m.streaming && m.content.trim())
    .map((m) => ({ role: m.role, content: m.content }))
}

async function sendMessage() {
  const question = input.value.trim()
  if (!question || loading.value) return

  const history = buildHistory()

  messages.value.push({ id: uid(), role: 'user', content: question })
  input.value = ''

  const assistantId = uid()
  messages.value.push({
    id: assistantId,
    role: 'assistant',
    content: '',
    citations: [],
    streaming: true,
    isLegal: undefined,
  })
  loading.value = true
  stickToBottom.value = true
  await scrollToBottom(true)

  try {
    const res = await fetch('/api/ask/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history }),
    })

    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || '请求失败')
    }
    if (!res.body) throw new Error('浏览器不支持流式响应')

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const parts = buffer.split('\n\n')
      buffer = parts.pop() ?? ''

      for (const part of parts) {
        const parsed = parseSseBlock(part)
        if (!parsed) continue

        const msg = messages.value.find((m) => m.id === assistantId)
        if (!msg) continue

        if (parsed.event === 'meta') {
          const { is_legal } = JSON.parse(parsed.data) as { is_legal: boolean }
          msg.isLegal = is_legal
        } else if (parsed.event === 'start') {
          const { status } = JSON.parse(parsed.data) as { status?: string }
          if (status === 'retrieving') msg.phase = 'retrieving'
          else if (status === 'verifying') msg.phase = 'verifying'
        } else if (parsed.event === 'citations') {
          msg.citations = JSON.parse(parsed.data) as Citation[]
        } else if (parsed.event === 'token') {
          const { content } = JSON.parse(parsed.data) as { content: string }
          msg.content += content
          msg.phase = undefined
          await scrollToBottom()
        } else if (parsed.event === 'answer_revision') {
          const payload = JSON.parse(parsed.data) as {
            content: string
            action?: CitationVerifyInfo['action']
          }
          msg.content = payload.content
          msg.answerRevised = true
          await scrollToBottom()
        } else if (parsed.event === 'done') {
          const payload = JSON.parse(parsed.data) as {
            disclaimer?: string
            is_legal?: boolean
            citation_verified?: boolean
            citation_verify?: CitationVerifyInfo
          }
          if (payload.disclaimer) disclaimer.value = payload.disclaimer
          if (payload.is_legal !== undefined) msg.isLegal = payload.is_legal
          if (payload.citation_verified !== undefined) {
            msg.citationVerified = payload.citation_verified
          }
          if (payload.citation_verify) {
            msg.citationVerify = payload.citation_verify
            msg.citationVerified = payload.citation_verify.citation_verified
          }
          msg.phase = undefined
          msg.streaming = false
        } else if (parsed.event === 'error') {
          const { message } = JSON.parse(parsed.data) as { message: string }
          throw new Error(message)
        }
      }
    }

    const msg = messages.value.find((m) => m.id === assistantId)
    if (msg) msg.streaming = false
  } catch (err) {
    messages.value = messages.value.filter((m) => m.id !== assistantId)
    ElMessage.error(err instanceof Error ? err.message : '请求失败')
  } finally {
    loading.value = false
    await scrollToBottom()
  }
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
}

function clearChat() {
  if (loading.value) return
  messages.value = []
  stickToBottom.value = true
}

const hasMessages = computed(() => messages.value.length > 0)

onMounted(loadLaws)
</script>

<template>
  <div class="layout">
    <header class="header">
      <div class="header-main">
        <div class="brand">
          <span class="logo">⚖</span>
          <div>
            <h1>法律智能问答</h1>
            <p>宪法 · 民法典 · 刑法 · 劳动法 RAG 检索 · DeepSeek</p>
          </div>
        </div>
        <el-button v-if="hasMessages" text type="primary" @click="clearChat">
          新对话
        </el-button>
      </div>
      <div v-if="laws.length" class="law-tags">
        <a
          v-for="law in laws"
          :key="law.id"
          :href="law.source_url"
          target="_blank"
          rel="noopener"
        >
          {{ law.name }}（{{ law.article_count }} 条）
        </a>
      </div>
    </header>

    <main class="chat-card">
      <div ref="chatBodyRef" class="chat-body" @scroll="onChatScroll">
        <div v-if="!hasMessages" class="welcome">
          <div class="welcome-icon">⚖</div>
          <h2>您好，我是法律智能助手</h2>
          <p>我会基于宪法、民法典、刑法、劳动法检索相关法条，并引用具体条款为您解答。</p>
          <div class="hints">
            <button type="button" @click="input = '结婚需要满足什么条件？'">
              结婚需要满足什么条件？
            </button>
            <button type="button" @click="input = '公民的基本权利有哪些？'">
              公民的基本权利有哪些？
            </button>
            <button type="button" @click="input = '合同无效的情形有哪些？'">
              合同无效的情形有哪些？
            </button>
          </div>
        </div>

        <div
          v-for="msg in messages"
          :key="msg.id"
          class="message-row"
          :class="msg.role"
        >
          <div class="avatar">{{ msg.role === 'user' ? '我' : 'AI' }}</div>
          <div class="bubble-wrap">
            <div class="bubble" :class="{ streaming: msg.streaming }">
              <div
                v-if="msg.content"
                class="bubble-text"
                v-html="renderContent(msg)"
              />
              <span v-else-if="msg.streaming" class="typing">{{ typingHint(msg) }}</span>
              <span
                v-if="msg.streaming && msg.content && msg.phase !== 'verifying'"
                class="cursor"
              >|</span>
              <span v-if="msg.streaming && msg.phase === 'verifying'" class="typing verify-hint">
                {{ typingHint(msg) }}
              </span>
            </div>

            <div
              v-if="msg.role === 'assistant' && !msg.streaming && msg.isLegal !== false && (msg.citationVerify || msg.citationVerified !== undefined)"
              class="verify-banner"
            >
              <div class="verify-head">
                <el-tag :type="verifyTagType(msg)" size="small" effect="light">
                  {{ verifyTagLabel(msg) }}
                </el-tag>
                <span v-if="verifySummary(msg)" class="verify-summary">
                  {{ verifySummary(msg) }}
                </span>
              </div>
              <ul
                v-if="msg.citationVerify?.invalid.length"
                class="verify-issues invalid"
              >
                <li v-for="(item, idx) in msg.citationVerify.invalid" :key="'i' + idx">
                  {{ formatCitationIssue(item) }}
                </li>
              </ul>
              <ul
                v-if="msg.citationVerify?.warnings.length"
                class="verify-issues warning"
              >
                <li v-for="(item, idx) in msg.citationVerify.warnings" :key="'w' + idx">
                  {{ formatCitationIssue(item) }}
                </li>
              </ul>
            </div>

            <div
              v-if="msg.role === 'assistant' && !msg.streaming && msg.isLegal !== false && msg.citations?.length"
              class="citations"
            >
              <div class="cite-title">引用法条</div>
              <el-collapse accordion>
                <el-collapse-item
                  v-for="(item, idx) in msg.citations"
                  :key="idx"
                  :title="`《${item.law_name}》${item.article_no}`"
                >
                  <p v-if="item.hierarchy" class="hierarchy">{{ item.hierarchy }}</p>
                  <p class="cite-text">{{ item.text }}</p>
                </el-collapse-item>
              </el-collapse>
            </div>
          </div>
        </div>
      </div>

      <div class="composer">
        <el-input
          v-model="input"
          type="textarea"
          :rows="2"
          :autosize="{ minRows: 2, maxRows: 5 }"
          placeholder="输入法律问题，Enter 发送，Shift+Enter 换行"
          maxlength="500"
          :disabled="loading"
          @keydown="onKeydown"
        />
        <el-button
          type="primary"
          circle
          :icon="Promotion"
          :loading="loading"
          :disabled="!input.trim()"
          @click="sendMessage"
        />
      </div>
    </main>

    <footer class="footer">
      <el-alert :title="disclaimer" type="warning" :closable="false" show-icon />
    </footer>
  </div>
</template>

<style scoped>
.layout {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: linear-gradient(160deg, #eef3f8 0%, #f8fafc 45%, #edf2f7 100%);
}

.header {
  padding: 16px 20px 12px;
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid #e4e7ed;
}

.header-main {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo {
  width: 42px;
  height: 42px;
  display: grid;
  place-items: center;
  border-radius: 12px;
  background: #1a3a5c;
  color: #fff;
  font-size: 20px;
}

.brand h1 {
  margin: 0;
  font-size: 20px;
  color: #1a3a5c;
}

.brand p {
  margin: 4px 0 0;
  font-size: 13px;
  color: #909399;
}

.law-tags {
  margin-top: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.law-tags a {
  font-size: 12px;
  color: #409eff;
  text-decoration: none;
  background: #ecf5ff;
  padding: 4px 10px;
  border-radius: 999px;
}

.chat-card {
  flex: 1;
  min-height: 0;
  margin: 12px 16px 0;
  background: #fff;
  border: 1px solid #e4e7ed;
  border-radius: 16px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: 0 8px 24px rgba(26, 58, 92, 0.06);
  position: relative;
}

.chat-body {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  overscroll-behavior: contain;
}

.welcome {
  max-width: 520px;
  margin: 8vh auto 0;
  text-align: center;
  color: #606266;
}

.welcome-icon {
  font-size: 48px;
  margin-bottom: 12px;
}

.welcome h2 {
  margin: 0 0 8px;
  color: #1a3a5c;
}

.welcome p {
  margin: 0;
  line-height: 1.7;
}

.hints {
  margin-top: 24px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.hints button {
  border: 1px solid #dcdfe6;
  background: #f5f7fa;
  border-radius: 10px;
  padding: 10px 14px;
  cursor: pointer;
  color: #303133;
  transition: 0.2s;
}

.hints button:hover {
  border-color: #409eff;
  color: #409eff;
  background: #ecf5ff;
}

.message-row {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
}

.message-row.user {
  flex-direction: row-reverse;
}

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  font-size: 13px;
  font-weight: 600;
  flex-shrink: 0;
}

.message-row.user .avatar {
  background: #409eff;
  color: #fff;
}

.message-row.assistant .avatar {
  background: #1a3a5c;
  color: #fff;
}

.bubble-wrap {
  max-width: min(720px, 78%);
}

.message-row.user .bubble-wrap {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
}

.bubble {
  position: relative;
  padding: 12px 16px;
  border-radius: 14px;
  line-height: 1.75;
}

.message-row.user .bubble {
  background: #409eff;
  color: #fff;
  border-bottom-right-radius: 4px;
}

.message-row.assistant .bubble {
  background: #f4f6f8;
  color: #303133;
  border-bottom-left-radius: 4px;
}

.bubble-text {
  font-size: 15px;
  word-break: break-word;
}

.bubble-text :deep(strong) {
  color: #1a3a5c;
  font-weight: 700;
}

.bubble-text :deep(.section-title) {
  display: inline-block;
  margin-top: 8px;
  color: #409eff;
}

.message-row.user .bubble-text :deep(strong) {
  color: #fff;
}

.typing {
  color: #909399;
  font-size: 14px;
}

.cursor {
  animation: blink 1s step-end infinite;
  color: #409eff;
  font-weight: 700;
}

@keyframes blink {
  50% {
    opacity: 0;
  }
}

.citations {
  margin-top: 10px;
  width: 100%;
}

.verify-banner {
  margin-top: 8px;
  padding: 10px 12px;
  border-radius: 10px;
  background: #f8fafc;
  border: 1px solid #ebeef5;
  width: 100%;
}

.verify-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.verify-summary {
  font-size: 12px;
  color: #909399;
}

.verify-issues {
  margin: 8px 0 0;
  padding-left: 18px;
  font-size: 12px;
  line-height: 1.6;
}

.verify-issues.invalid {
  color: #f56c6c;
}

.verify-issues.warning {
  color: #e6a23c;
}

.verify-hint {
  display: block;
  margin-top: 6px;
}

.cite-title {
  font-size: 13px;
  color: #909399;
  margin-bottom: 6px;
}

.hierarchy {
  color: #909399;
  margin: 0 0 8px;
  font-size: 13px;
}

.cite-text {
  margin: 0;
  line-height: 1.7;
  font-size: 14px;
}

.composer {
  display: flex;
  gap: 10px;
  align-items: flex-end;
  padding: 14px 16px;
  border-top: 1px solid #ebeef5;
  background: #fafbfc;
}

.composer .el-textarea {
  flex: 1;
}

.footer {
  padding: 12px 16px 16px;
}
</style>

<style>
html,
body,
#app {
  height: 100%;
  margin: 0;
}

body {
  font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}
</style>
