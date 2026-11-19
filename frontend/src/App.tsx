// Walking skeleton: shows that the frontend reaches the API and the database.
// The real screens (login, karts, chat, ...) come in their own issues, under src/features/.
import { useEffect, useState } from 'react'
import { ApiError, api } from './lib/api'

type Health = { status: string; database: string }

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [error, setError] = useState<ApiError | null>(null)

  useEffect(() => {
    api<Health>('/health')
      .then(setHealth)
      .catch((e: ApiError) => setError(e))
  }, [])

  return (
    <main>
      <h1>MechAi</h1>
      <p>Repair help for the Supercube workshop.</p>
      <div className="status">
        {health && <span className="ok">API and database are running</span>}
        {error && (
          <span className="bad">
            {error.message} <small>({error.code})</small>
          </span>
        )}
        {!health && !error && <span>Checking the API…</span>}
      </div>
      <p>
        <a href="/api/docs">Open the API docs</a> to try the endpoints.
      </p>
    </main>
  )
}
