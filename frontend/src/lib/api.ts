// One helper for every API call. It reads the error shape of the backend
// ({"error": {"code", "message", "reference"}}) so screens can show a translated message.

// Error thrown by api(): the backend code, a readable message and the request reference.
export class ApiError extends Error {
  code: string
  reference: string | null

  constructor(code: string, message: string, reference: string | null) {
    super(message)
    this.code = code
    this.reference = reference
  }
}

// Call the backend at /api/v1 + path and return the JSON body.
// Throws ApiError when the server is offline, the body is not JSON, or the status is an error.
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers)
  if (!headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  let response: Response
  try {
    response = await fetch(`/api/v1${path}`, {
      ...init,
      headers,
    })
  } catch {
    throw new ApiError('CORE_OFFLINE', 'The server cannot be reached.', null)
  }
  // Read the body; an error response without JSON still gets a generic error below.
  type Json = null | boolean | number | string | Json[] | { [key: string]: Json }
  let body: Json = null
  try {
    body = await response.json() as Json
  } catch {
    if (response.ok) {
      throw new ApiError('CORE_INVALID_RESPONSE', 'The server returned an invalid response.', null)
    }
  }
  // Take the error fields from the body, with safe defaults when they are missing.
  if (!response.ok) {
    const value = body && typeof body === 'object' && !Array.isArray(body) ? body.error : null
    const error = value && typeof value === 'object' && !Array.isArray(value) ? value : null
    throw new ApiError(
      typeof error?.code === 'string' ? error.code : 'CORE_INTERNAL',
      typeof error?.message === 'string' ? error.message : 'Something went wrong.',
      typeof error?.reference === 'string' ? error.reference : null,
    )
  }
  return body as T
}
