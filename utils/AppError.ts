export class AppError extends Error {
  constructor(public message: string, public code: string, public cause?: unknown) {
    super(message);
    this.name = 'AppError';
  }
}

export class ParsingError extends AppError {
  constructor(message: string, cause?: unknown) {
    super(message, 'PARSING_ERROR', cause);
  }
}

export class AnalysisError extends AppError {
  constructor(message: string, cause?: unknown) {
    super(message, 'ANALYSIS_ERROR', cause);
  }
}
