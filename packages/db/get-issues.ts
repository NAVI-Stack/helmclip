import postgres from "postgres";
import { resolveDatabaseTarget } from "./src/runtime-config.ts";

async function main() {
  const target = resolveDatabaseTarget();
  let connStr = target.mode === "postgres" ? target.connectionString : `postgres://paperclip:paperclip@127.0.0.1:${target.port}/paperclip`;
  
  const sql = postgres(connStr);
  const runs = await sql`
    SELECT id, status, error, error_code, invocation_source, result_json, liveness_state, liveness_reason
    FROM heartbeat_runs 
    WHERE status = 'failed' OR error IS NOT NULL OR error_code IS NOT NULL
    ORDER BY created_at DESC 
    LIMIT 20
  `;

  console.log("Recent Failed Runs:");
  for (const row of runs) {
    console.log(`Run: ${row.id}`);
    console.log(`Status: ${row.status}`);
    console.log(`Error: ${row.error_code} - ${row.error}`);
    console.log(`Invocation: ${row.invocation_source}`);
    console.log(`Liveness: ${row.liveness_state} - ${row.liveness_reason}`);
    console.log(`Result JSON: ${JSON.stringify(row.result_json)?.slice(0, 200)}`);
    console.log("---");
  }

  process.exit(0);
}

main().catch(console.error);
