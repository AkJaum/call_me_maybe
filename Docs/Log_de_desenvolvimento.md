# Log de desenvolvimento — Call Me Maybe

Este documento registra a evolução do projeto em linguagem de estudo: o que foi
feito, por que foi feito e como cada parte funciona. O `en.subject.pdf` versão
1.5 permanece como fonte primária; este log não substitui o subject.

## Estado encontrado antes da retomada

O repositório já possuía uma fundação organizada, mas ainda não executava
*function calling*. Existiam:

- configuração Python 3.10+ com `pyproject.toml`, `uv.lock` e pacote local
  `llm_sdk`;
- regras de Makefile para instalação, execução, depuração, limpeza, testes e
  lint;
- modelos Pydantic básicos para definições, prompts e resultados;
- leitura e escrita JSON com mensagens de erro legíveis;
- CLI com caminhos padrão e modos auxiliares;
- adaptador inicial para as operações públicas do SDK;
- dois exemplos de funções, dois prompts e um teste simples de leitura.

O comando principal ainda encerrava com uma mensagem de scaffold. Não havia
schema dinâmico completo, gramática JSON, integração com vocabulário, máscara de
logits nem geração token a token.

## Etapa 1 — Contratos de schema

### O que foi feito

- O catálogo passou a exigir pelo menos uma função e nomes únicos.
- Os tipos escalares suportados foram fechados em `string`, `number`, `integer`
  e `boolean`.
- `array` e `object` passaram a ser recusados explicitamente enquanto o suporte
  complexo, listado como bônus no subject, não estiver implementado.
- O resultado passou a ser validado dinamicamente contra a função escolhida.
- Nome desconhecido, argumentos ausentes ou extras e tipos errados são rejeitados.
- Booleanos não são aceitos como inteiros/números, apesar da herança interna de
  `bool` em Python.
- `NaN` e infinitos são rejeitados porque não são números JSON válidos.
- `QwenClient` foi migrado de `dataclass` comum para `BaseModel` Pydantic.

### Por que

O subject exige que todas as saídas respeitem exatamente o schema recebido e que
as classes usem Pydantic. Apenas declarar `parameters: dict[str, Any]` garantia a
forma externa, mas não ligava o resultado à função selecionada. O catálogo também
precisava impedir duas funções com o mesmo nome, pois isso tornaria a escolha
ambígua.

### Como funciona

`FunctionCatalog` valida a coleção inteira. `build_function_call_result` fornece
as definições como contexto ao `FunctionCallResult`; o validator encontra a
função pelo nome, compara exatamente o conjunto de chaves e testa cada valor com
o tipo declarado. Assim, a validação final é independente da confiança no texto
produzido pelo modelo.

## Etapa 2 — Gramática incremental

### O que foi feito

Foi criado `src/grammar.py`, que representa os estados `invalid`, `prefix` e
`complete` de um documento canônico:

```json
{"name":"funcao_declarada","parameters":{"argumento":"valor"}}
```

A gramática:

- mantém múltiplas funções possíveis enquanto o prefixo ainda é ambíguo;
- limita nomes e chaves ao catálogo;
- exige todos os argumentos na ordem canônica e proíbe extras;
- valida strings, escapes JSON, Unicode, números, inteiros e booleanos;
- aceita funções sem argumentos;
- rejeita texto depois do fechamento;
- aceita fragmentos inteiros com `can_accept`, não apenas caracteres;
- produz um novo estado imutável com `advance`.

### Por que

O constrained decoding precisa decidir, a cada token, se o fragmento textual
inteiro daquele token ainda pode terminar em JSON válido e compatível com algum
schema. Um `json.loads` aplicado somente no fim detectaria o erro tarde demais. A
gramática incremental impede que o prefixo inválido seja gerado.

O `prompt` original não é gerado pelo LLM porque o programa já o conhece. Ele será
anexado deterministicamente ao resultado final, que continuará contendo
exatamente `prompt`, `name` e `parameters`, como exige o subject.

### Como funciona

Para cada função ainda possível, a gramática compara a estrutura fixa e analisa o
valor conforme seu tipo. Um fragmento é aceito se ao menos uma definição puder
continuar válida. Quando todo o documento fecha exatamente, o estado se torna
`complete` e nenhum conteúdo adicional pode entrar.

## Verificação das etapas 1 e 2

- 18 testes passaram;
- `flake8 .` passou;
- `mypy` passou com as flags obrigatórias do subject;
- a CLI em `--validate-only` validou os dados de demonstração;
- `git diff --check` não encontrou whitespace inválido.

## Etapa 3 — Vocabulário e tokens

### O que foi feito

Foi criado `src/vocabulary.py` com `TokenVocabulary`. Ele:

- recebe o caminho retornado por `get_path_to_vocab_file`, método público do SDK;
- lê e valida o `vocab.json` sem acessar tokenizer ou atributos privados;
- exige o formato `token textual -> ID inteiro`, IDs não negativos e únicos;
- converte os símbolos do byte-level BPE usado pelo Qwen em bytes reais;
- decodifica esses bytes como UTF-8;
- associa cada ID utilizável ao fragmento textual que ele acrescenta;
- filtra IDs fora do tamanho real do vetor de logits;
- passa o fragmento inteiro pela gramática, pois um token pode conter várias
  letras e sinais estruturais ao mesmo tempo;
- guarda em cache os IDs permitidos para a combinação de schema, prefixo e
  tamanho dos logits.

### Por que

O subject chama atenção para o fato de que tokens não são palavras nem
caracteres. O Qwen usa marcadores byte-level: por exemplo, `Ġhello` representa
` hello`, e `Ã§` representa os bytes UTF-8 de `ç`. Comparar diretamente os textos
do `vocab.json` com a gramática faria espaços e Unicode serem interpretados de
forma errada.

Alguns tokens representam somente uma parte de uma sequência UTF-8. Foram
excluídos os que não decodificam isoladamente. Isso mantém a propriedade de que
todo token aceito acrescenta texto Unicode válido. Unicode ainda pode ser gerado
por tokens completos ou por escapes JSON ASCII como `\u00e7`. Nomes de funções e
parâmetros são serializados de forma ASCII-canônica para não depender de tokens
UTF-8 parciais na estrutura obrigatória.

### Como funciona

`_byte_decoder` reconstrói a tabela inversa popularizada pelo tokenizer GPT-2 e
usada pelo BPE do Qwen. Cada caractere artificial volta ao byte original, e o
conjunto de bytes do token vira o fragmento real. `allowed_token_ids` testa:

```text
ID dentro dos logits + fragmento UTF-8 válido + grammar.can_accept(fragmento)
```

Somente os IDs que satisfazem as três condições chegam à máscara da geração.

### Verificação real

Com `Qwen/Qwen3-0.6B`:

- quantidade de logits: 151.936;
- tokens UTF-8 utilizáveis: 150.195;
- tokens incompletos ignorados: 1.448;
- maior ID útil encontrado: 151.642, dentro do vetor de logits;
- exemplos confirmados: ID `90 -> "{"` e ID `220 -> " "`.

## Etapa 4 — Geração restringida

### O que foi feito

Foi criado `src/generation.py` com:

- `build_model_prompt`, que apresenta funções e pedido ao modelo;
- `mask_invalid_logits`, que coloca `-inf` em todo token inválido;
- `select_highest_logit`, que escolhe o maior logit ainda permitido;
- `GenerationConfig`, com limite validado de novos tokens;
- `ConstrainedDecoder`, que executa o ciclo completo token a token;
- parsing Pydantic e validação dinâmica final do documento completo;
- erros controlados para logits vazios, ausência de continuação válida, saída
  final inválida e limite excedido.

### Por que

O requisito central do subject é que o LLM escolha função e valores, mas não tenha
permissão para quebrar JSON/schema. Por isso não existe busca de palavras no
prompt, regex para escolher função nem tabela de respostas. O modelo produz os
logits; a gramática define apenas o conjunto permitido; e o maior logit desse
conjunto vence.

Usar `-inf` explicitamente reproduz a operação descrita pelo subject. A seleção
greedy foi escolhida para tornar a execução determinística e facilitar testes e
reprodução. Se o modelo der a maior nota global a um token inválido, esse token é
mascarado e não pode ser escolhido.

### Como funciona

Para cada prompt:

1. as definições são serializadas e incluídas no contexto do modelo;
2. o prompt completo é convertido em IDs pelo `encode` público;
3. a gramática começa vazia e mantém todas as funções possíveis;
4. `get_logits_from_input_ids` retorna as notas do próximo token;
5. o vocabulário encontra os IDs compatíveis com a gramática;
6. todos os demais logits viram `-inf`;
7. o maior logit restante fornece o próximo ID;
8. seu fragmento avança a gramática e seu ID é anexado ao contexto;
9. o ciclo termina somente quando o JSON está completo ou quando o limite gera um
   erro controlado;
10. `json.loads`, `GeneratedFunctionCall` e `FunctionCallResult` validam novamente
    a saída; então o prompt original é anexado deterministicamente.

### Teste controlado

O SDK falso deu nota `100` a um token `#` inválido e nota `10` ao próximo token
correto. A geração bloqueou `#` em todas as etapas e produziu:

```json
{"prompt":"Add 2 and 3","name":"fn_add","parameters":{"a":2,"b":3}}
```

Isso demonstra que a escolha válida continua vindo dos logits, mas a estrutura
não depende da boa vontade do modelo.

### Testes reais com Qwen3-0.6B

Os dois exemplos atuais foram executados com o modelo real e a API pública do
SDK:

```json
{"prompt":"What is the sum of 2 and 3?","name":"fn_add_numbers","parameters":{"a":2,"b":3}}
{"prompt":"Greet Shrek","name":"fn_greet","parameters":{"name":"Shrek"}}
```

Ambos escolheram a função correta, extraíram os argumentos com tipos corretos e
terminaram em JSON compatível com o schema.

### Limites conhecidos desta fase

- Naquele momento, a CLI ainda não processava a lista inteira nem gravava o
  arquivo final. Isso foi resolvido na etapa 5.
- O SDK público recalcula a sequência completa a cada token e não expõe cache de
  atenção. Em CPU, isso domina o tempo da geração.
- O filtro inicial percorre muitos tokens do vocabulário; o cache evita repetir o
  mesmo estado, mas otimizações adicionais deverão ser medidas antes da entrega.
- A acurácia de 90%+ ainda precisa de um conjunto rotulado maior; dois exemplos
  corretos validam integração, não a métrica final.

## Verificação acumulada após as etapas 3 e 4

- 24 testes passaram na revisão final;
- teste específico prova a máscara `-inf`;
- testes cobrem vocabulário byte-level, Unicode, entradas inválidas e limite;
- integração real de vocabulário, encode e logits passou;
- duas gerações reais do Qwen produziram chamadas corretas;
- `flake8` passou;
- `mypy` passou com todas as flags obrigatórias do subject;
- `python -m src --validate-only` passou;
- compilação sintática e `git diff --check` passaram.

## Etapa 5 — Pipeline completo da CLI

### O que foi feito

- Foi criado `src/pipeline.py` para gerar um resultado validado para cada prompt,
  preservando a ordem de entrada.
- A CLI agora carrega entradas, inicializa o Qwen uma vez, carrega um vocabulário,
  cria um decoder e reutiliza todos eles no lote inteiro.
- O resultado só é gravado depois que todos os prompts terminam corretamente.
- `--functions_definition`, `--input` e `--output` funcionam com caminhos padrão
  ou personalizados.
- Um array de prompts vazio produz `[]` sem carregar o modelo.
- Erros de entrada, SDK, vocabulário, geração e saída são convertidos em mensagem
  legível no `stderr` e código de saída 1.
- Uma proteção final captura falhas inesperadas no limite da CLI para cumprir a
  regra de não encerrar com traceback para o usuário.
- `make run` e `make debug` deixaram de acrescentar `--validate-only` e agora
  executam o programa real.

### Por que

Carregar o modelo para cada prompt repetiria o maior custo do programa e impediria
o reaproveitamento do cache de tokens permitidos. Acumular todos os resultados
antes da escrita também evita que uma falha no meio do lote produza um arquivo
parcial que pareça válido.

O lote vazio é um caso válido e determinístico: não há trabalho para o LLM, mas o
resultado JSON correto continua sendo um array vazio.

### Como funciona

O fluxo normal agora é:

```text
CLI
 ├─ valida catálogo e prompts
 ├─ carrega Qwen uma vez
 ├─ carrega vocabulário uma vez
 ├─ gera e valida cada chamada em ordem
 ├─ acumula todas as chamadas
 └─ grava o array JSON atomicamente
```

`write_results` cria um arquivo temporário no mesmo diretório da saída, serializa,
faz `flush` e `fsync` e usa `os.replace`. Se a escrita ou substituição falhar, o
temporário é removido e o arquivo anterior permanece intacto. Como a troca ocorre
no mesmo filesystem, não existe janela em que o destino contenha JSON pela
metade.

### Teste real ponta a ponta

A CLI foi executada com os dois arquivos padrão, uma única carga do
`Qwen/Qwen3-0.6B` e um caminho de saída em `/tmp`. Ela terminou em aproximadamente
2min38s, abaixo do limite de 5 minutos do subject, e escreveu:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2, "b": 3}
  },
  {
    "prompt": "Greet Shrek",
    "name": "fn_greet",
    "parameters": {"name": "Shrek"}
  }
]
```

O arquivo foi relido com `json.loads`, teve dois objetos e cada objeto continha
exatamente `prompt`, `name` e `parameters`.

## Etapa 6 — Ampliação dos testes

### O que foi feito

A suíte passou de 24 para 47 testes. A cobertura comportamental agora inclui:

- arquivos ausentes e JSON malformado com localização;
- raiz de tipo incorreto;
- prompt ausente, vazio ou com campo extra;
- catálogo vazio ou com nomes duplicados;
- funções e tipos não suportados;
- parâmetros ausentes, extras ou de tipo errado;
- booleano versus inteiro/número e floats não finitos;
- strings vazias, Unicode, escapes, barras e aspas;
- inteiros grandes, números negativos, decimais e expoentes;
- funções alternativas e funções sem argumentos;
- estrutura, ordem, nomes e conteúdo extra inválidos;
- vocabulário byte-level, tokens UTF-8 incompletos, IDs negativos/duplicados e
  tokens fora dos logits;
- máscara `-inf`, logits vazios/não finitos e ausência de continuação;
- limite máximo de geração e validação final redundante;
- CLI com caminhos personalizados e preservação da ordem;
- reutilização de um decoder para todo o lote;
- lote vazio sem carga do modelo;
- falha no meio do lote sem sobrescrever uma saída anterior;
- escrita Unicode e remoção do temporário após falha atômica.

### Por que

As principais propriedades deste projeto são negativas: certas saídas jamais
podem acontecer. Por isso os testes não verificam somente exemplos corretos; eles
tentam violar cada fronteira e confirmam que a execução falha de forma controlada.
O SDK falso torna possível testar logits e máscaras rapidamente, enquanto os
testes reais confirmam que as premissas correspondem ao Qwen fornecido.

### Verificação final das etapas 5 e 6

- `make test`: 47 testes passaram;
- `make lint`: `flake8` e `mypy` com todas as flags obrigatórias passaram;
- `git diff --check`: passou;
- CLI real com dois prompts: passou;
- saída real relida e validada: passou;
- duração real observada: aproximadamente 2min38s para o lote demonstrativo.

## Etapa 7 — Qualidade, acurácia e desempenho

### O que foi feito

- Foi criado um conjunto diagnóstico rotulado em `benchmarks/cases.json`.
- Foi criado `src/benchmark.py`, executável por `make benchmark`.
- O benchmark mede validade JSON, validade de schema, acerto de função, acerto de
  argumentos, acerto completo, duração total e pico de memória residente.
- Os rótulos são usados somente depois da geração para pontuar; não entram no
  prompt, na gramática nem no programa principal.
- O relatório é salvo em `benchmarks/latest_results.json`.
- O comando retorna 0 somente se atingir simultaneamente 100% de validade, pelo
  menos 90% de acerto de função/argumentos e menos de 300 segundos.
- Foram adicionados três testes do carregamento e cálculo das métricas.

### Casos escolhidos

Os quatro casos complementam os dois exemplos fornecidos:

- inteiro negativo grande e zero;
- dois números decimais, incluindo negativo;
- sinônimo menos direto de saudação com Unicode;
- saudação com nome hifenizado.

Isso não transforma quatro casos em evidência estatística ampla. O objetivo é
ter uma régua local reproduzível que detecte regressões e exponha limites.

### Primeira medição

O primeiro benchmark real produziu:

- JSON válido: 100%;
- schema válido: 100%;
- função correta: 4/4 (100%);
- argumentos corretos: 4/4 (100%);
- acerto completo: 4/4 (100%);
- tempo: 355,596s;
- pico de memória: 5120,086 MiB;
- metas conjuntas do subject: não atingidas por tempo.

### Otimização investigada e removida

Foi testado o pré-preenchimento de caracteres estruturalmente inevitáveis. A ideia
era evitar logits para `{`, aspas, chaves e prefixos comuns de funções, mantendo
bifurcações e valores sob decisão do LLM.

A implementação estava correta e testada, mas exigia varrer 150 mil tokens a cada
caractere obrigatório. Na repetição real, o primeiro caso ainda não havia
terminado após mais de 210s. A execução foi interrompida e a alteração removida.
Ela não permaneceu no projeto porque a medição provou regressão.

### Otimização mantida

Foi medido que as respostas corretas usam 14–25 tokens, enquanto os prompts do
modelo tinham 108–118 tokens. Como o SDK recalcula todo o contexto em cada passo,
a prosa redundante do prompt foi condensada sem remover nome, descrição,
parâmetros, tipos ou pedido. O caso mais pesado caiu de 118 para 90 tokens de
contexto inicial.

Na repetição final:

- JSON/schema: 100%;
- função: 4/4 (100%);
- argumentos: 4/4 (100%);
- tempo: 348,502s;
- pico de memória: 5120,520 MiB;
- redução observada: 7,094s;
- meta de 300s do benchmark ampliado: ainda não atingida.

O lote padrão de dois prompts havia terminado em aproximadamente 158s e continua
abaixo dos cinco minutos. O README distingue claramente esse resultado do
benchmark diagnóstico mais pesado. `make benchmark` retorna 1 na medição gravada,
como deve, em vez de esconder a falha de tempo.

### Por que o gargalo permanece

O método público `get_logits_from_input_ids` recebe toda a sequência e não expõe
cache de atenção. Portanto, cada novo token executa novamente o modelo sobre o
prompt e todos os tokens anteriores. O cache implementado evita repetir filtros
de vocabulário para o mesmo schema/prefixo, mas não pode eliminar o custo interno
do modelo sem usar API privada, o que o subject proíbe.

## Etapa 8 — README obrigatório

### O que foi feito

O `README.md` foi escrito em inglês e começa com a frase curricular exigida,
usando o login `akjaum`. Ele contém:

- descrição, objetivo e exemplo;
- requisitos, instalação e execução;
- formatos completos de entrada e saída;
- explicação detalhada do constrained decoding;
- arquitetura e responsabilidades dos arquivos;
- decisões de design;
- tratamento de erros;
- estratégia e escopo de testes;
- desempenho medido com distinção entre lote padrão e benchmark;
- desafios e soluções;
- limitações conhecidas;
- referências primárias;
- descrição transparente de como IA foi usada.

### Fontes consultadas

Foram priorizadas fontes oficiais: RFC 8259, documentação Python de JSON,
Pydantic, uv, documentação de tokenização do Hugging Face, model card do Qwen e o
relatório técnico Qwen3.

### Divergência observada na régua

A régua complementar menciona chaves `prompt`, `fn_name` e `args`, enquanto o
subject v1.5 determina explicitamente `prompt`, `name` e `parameters` e fornece
exemplos com essas chaves. O projeto mantém `name` e `parameters` porque o subject
é a fonte primária definida para o desenvolvimento. Essa divergência deve ser
explicada durante a defesa caso o avaliador a levante.

### Verificação final das etapas 7 e 8

- 52 testes passaram, incluindo dois contratos automáticos do README;
- `flake8` passou;
- `mypy` passou com as flags obrigatórias;
- compilação de todos os arquivos Python passou;
- `git diff --check` passou;
- relatório do benchmark foi preservado em formato JSON reproduzível.

## Etapa 9 — Bônus isolados e demonstráveis

### O que foi feito

- Foi adicionada a opção `--visualize <arquivo.html>` e o alvo
  `make visualize`.
- A geração opcional registra, a cada passo, índice, ID escolhido, fragmento,
  quantidade de tokens permitidos, logit vencedor e prefixo válido acumulado.
- O relatório também mostra modelo, duração, resultado final e estatísticas de
  acerto/erro do cache de tokens permitidos.
- `TokenVocabulary` passou a expor uma fotografia Pydantic imutável dos contadores
  do cache sem permitir alteração externa do seu estado.
- O HTML é autônomo, escapa prompt e texto gerado e é escrito atomicamente.
- A rota normal continua sem armazenar os passos e produz exatamente o mesmo JSON
  obrigatório.
- Seis testes foram acrescentados, levando a suíte de 52 para 58 testes.

### Por que

“Visualization of the generation process”, otimizações por cache, mecanismos de
recuperação e suíte abrangente aparecem explicitamente como bônus no subject.
A visualização é especialmente útil na defesa porque permite mostrar que a
máscara atua em cada token: o avaliador consegue ver o conjunto permitido e a
decisão do logit sem precisar interpretar apenas o código.

O bônus foi colocado atrás de uma opção para não alterar formato, desempenho ou
interface obrigatória. Batching, modelos adicionais, tokenizer reimplementado e
argumentos aninhados não foram anunciados, pois bônus só contam quando funcionam
e podem ser demonstrados de ponta a ponta.

### Como funciona

`generate` e `generate_with_trace` usam o mesmo loop interno. O primeiro descarta
os detalhes; o segundo cria modelos `GenerationStep` e `GenerationTrace`. Depois
de toda a chamada ser validada, `src/visualization.py` transforma os registros em
uma tabela HTML. Todo conteúdo variável passa por escape antes de entrar no
documento e tanto o JSON quanto o HTML usam arquivo temporário seguido de
`os.replace`. A auditoria também encontrou e fechou uma lacuna de recuperação:
arquivos JSON ou vocabulários com bytes UTF-8 inválidos agora recebem erro de
domínio específico, em vez de depender da proteção genérica da CLI.

### Verificação da etapa 9

- 58 testes passaram;
- `flake8` passou;
- `mypy` passou com as flags obrigatórias;
- `git diff --check` passou;
- testes específicos verificam coleta do trace, hits do cache, escape de HTML,
  lote vazio e preservação do destino quando a escrita falha.
- `make run` real passou em 148,89s com dois resultados corretos;
- `make visualize` real passou em 145,26s e mostrou 5 hits de cache no segundo
  prompt;
- `make lint-strict` também passou;
- a auditoria final e as ressalvas para a peer evaluation foram registradas em
  `ESTADO_DO_PROJETO.md`.
