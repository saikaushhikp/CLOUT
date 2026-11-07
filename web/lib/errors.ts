/** Transport-level failure (HTTP 4xx/5xx or network). Game-rule rejections are NOT errors (accepted:false). */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;
  constructor(status: number, detail: unknown, message?: string) {
    super(message ?? (typeof detail === "string" ? detail : `HTTP ${status}`));
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}
