import llama_cpp
import torch

# Given a list of tokens and a list of tokens to be considered as markers, return indices of the marker tokens
def marker_tokens(x:list[str], markers:list[str]=['.', ',']) -> list[str]:
    return [index for index, token in enumerate(x) if token in markers]

# Cross-entropy loss function that adds variance of targets when current token is a marker token
def loss_function(x:torch.tensor, y:torch.tensor, marker:bool=False) -> torch.tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(x, y) + (0, torch.var(y))[marker]

# Given a prompt, split the LLM's answer into pieces based on whether the variance of the logits is within the radius
def sentence_piece(prompt:str, prompt_template:str="", model_path:str="", k:int=10, radius:float=0.02, verbose:bool=False, latex_table_rows:int=0, cut_off:int=-1, model_kwargs:dict={'temp': 0}) -> list[str]:

    # Initialize arrays
    current_piece = []
    pieces = []
    output = []
    variances = []

    # Load model and convert prompt into bytes
    model = llama_cpp.Llama(model_path)
    prompt = bytes(prompt_template.format(prompt=prompt), encoding='utf-8')

    # If verbose, print headers of token-variance table
    print("|Token|Variance|\n|---|")

    # Main loop #
    tokens = model.tokenize(prompt)
    for token in model.generate(tokens, **model_kwargs):
        
        # Get variance of top-k logits
        logits = torch.tensor(model.eval_logits)
        logits = torch.softmax(logits, dim=1)
        logits = torch.sort(logits)
        top_k_logits = logits.values[0][-k:]
        variance = torch.var(top_k_logits)

        # Get current token
        current_token = str(model.detokenize([token]), encoding='utf-8')

        # If verbose, print current token and current logit variance
        if verbose:
            print(f"|{current_token}|{float(variance):.2}|")

        # Update arrays
        output.append(current_token)
        current_piece.append(current_token)
        variances.append(variance)
        
        # If variance <= radius, store the current piece in pieces array, and clear the current piece
        if torch.le(variance, torch.tensor(radius)):
            pieces.append(current_piece)
            current_piece = []

        # Stop generation after reaching cut-off length or EOS token
        if token == model.token_eos() or (len(output) >= cut_off and cut_off > 0):
            
            # Add current piece to pieces array
            pieces.append(current_piece)
            current_piece = []

            # Join tokens in each piece into a line
            pieces = [''.join(piece) for piece in pieces]

            # If verbose, print pieces
            if verbose:
                print(f"Pieces: {pieces}")
            
            # If latex_table_rows is greater than zero, return both the pieces and LaTeX table describing the first latex_table_rows tokens and their variances
            if latex_table_rows > 0:
                variances = [f"{float(variance):.2}" for variance in variances]
                latex_table = ' & '.join(output[:latex_table_rows]) + '\\ \n' + ' & '.join(variances[:latex_table_rows]) + '\\\n'
                latex_table = r'\begin{table}[!th]' + '\n' + r'\begin{tabular}' + '\n' + latex_table + r'\caption{Prompt:' + str(prompt, encoding='utf-8') + '}' + '\n' + r'\end{tabular}' + '\n' + r'\end{table}'
                return pieces, latex_table
            
            # Else return only the pieces
            else:
                return pieces

# Example #
# sentence_piece("Write a very short haiku about rain", "[INST] <|user|>{prompt} [/INST]<|bot|>", "./models/mistral-7b-instruct-v0.1.Q3_K_M.gguf", verbose=True)