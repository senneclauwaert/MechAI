// One helper for every API call. It reads the error shape of the backend
// ({"error": {"code", "message", "reference"}}) so screens can show a translated message.

export class ApiError extends Error {
  code: string
  reference: string | null

  constructor(code: string, message: string, reference: string | null) {
    super(message)
    this.code = code
    this.reference = reference
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`/api/v1${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch {
    throw new ApiError('CORE_OFFLINE', 'The server cannot be reached.', null)
  }
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    const error = body?.error
    throw new ApiError(
      error?.code ?? 'CORE_INTERNAL',
      error?.message ?? 'Something went wrong.',
      error?.reference ?? null,
    )
  }
  return body as T
}
