export interface Token {
  id: number;
  text: string;
  special: boolean;
}

export interface Payload {
  token: Token;
  generated_text?: string;
  details?: boolean;
}
