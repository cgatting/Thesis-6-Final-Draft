export type Result<T, E = Error> = { ok: true; value: T } | { ok: false; error: E };

export const ok = <T>(value: T): Result<T, never> => ({ ok: true, value });
export const err = <E>(error: E): Result<never, E> => ({ ok: false, error });

export const unwrap = <T, E>(result: Result<T, E>): T => {
  if (result.ok) return result.value;
  throw result.error;
};

export const mapResult = <T, U, E>(result: Result<T, E>, fn: (val: T) => U): Result<U, E> => {
    if (result.ok) return ok(fn(result.value));
    return result as any;
};
