# Call Me Maybe — estado atual e plano de desenvolvimento

> Documento de diagnóstico elaborado a partir do `en.subject.pdf`, versão 1.5.
> O subject é tratado aqui como a fonte primária de requisitos. Este documento
> descreve o que existe no repositório e registra a auditoria local de entrega.

## 1. Objetivo definido pelo subject

O projeto deve implementar, em Python 3.10 ou superior, uma ferramenta de
*function calling*. Ela recebe:

1. um arquivo JSON com definições de funções; e
2. um arquivo JSON com pedidos em linguagem natural.

Para cada pedido, o programa deve usar o modelo `Qwen/Qwen3-0.6B` para escolher
a função adequada e extrair todos os seus argumentos. O resultado deve ser um
único array JSON cujos objetos contenham **exatamente**:

```json
{
  "prompt": "pedido original",
  "name": "nome_da_funcao",
  "parameters": {}
}
```

O ponto central do exercício não é pedir ao modelo que escreva JSON livremente.
A implementação deve fazer **constrained decoding token a token**: consultar os
logits do modelo, permitir somente tokens que mantenham o JSON e o schema da
função válidos e impedir os demais. A função deve ser escolhida pelo LLM, sem
heurísticas ou respostas codificadas a partir dos exemplos.

Metas obrigatórias ou esperadas no subject:

- 100% das saídas devem ser JSON válido e compatível com o schema;
- seleção de função e extração de argumentos com pelo menos 90% de acerto;
- processamento do conjunto de testes em menos de 5 minutos;
- erros de arquivos, JSON e casos extremos devem ser tratados sem encerramento
  inesperado;
- uso exclusivo da API pública de `llm_sdk`;
- classes validadas com Pydantic, código tipado, docstrings PEP 257, `flake8` e
  `mypy` sem erros;
- execução pela interface `uv run python -m src`, com os três caminhos opcionais;
- `README.md` completo e escrito em inglês.

## 2. Estrutura atualmente existente

```text
.
├── en.subject.pdf
├── pyproject.toml
├── uv.lock
├── Makefile
├── .flake8
├── .gitignore
├── Context.md
├── README.md                    # documentação obrigatória em inglês
├── Log_de_desenvolvimento.md    # guia cronológico em português
├── benchmarks/                  # casos rotulados e última medição
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── io.py
│   ├── model.py
│   ├── models.py
│   ├── generation.py
│   ├── grammar.py
│   ├── pipeline.py
│   ├── visualization.py
│   └── vocabulary.py
├── tests/
│   └── test_io.py
├── data/input/
│   ├── functions_definition.json
│   └── function_calling_tests.json
└── llm_sdk/
    ├── pyproject.toml
    ├── uv.lock
    └── llm_sdk/__init__.py
```

Também existem `data.zip` e `llm_sdk.zip`, aparentemente os arquivos originais
fornecidos com o projeto. O diretório `data/output/` está corretamente ignorado e
não está versionado, como determina o subject.

## 3. O que já foi estruturado

### 3.1 Gerenciamento do projeto

O `pyproject.toml`:

- define Python `>=3.10`;
- declara `numpy`, `pydantic` e o pacote local `llm-sdk`;
- referencia o SDK pelo caminho `llm_sdk` usando `[tool.uv.sources]`;
- separa `flake8`, `mypy` e `pytest` como dependências de desenvolvimento;
- configura descoberta de testes e exclui o SDK fornecido da análise do `mypy`.

O `uv.lock` está presente. Isso está alinhado com a forma de instalação que será
usada pela avaliação (`uv sync`).

### 3.2 Automação

O `Makefile` possui as regras obrigatórias `install`, `run`, `debug`, `clean` e
`lint`, além de `lint-strict`, `test`, `model-check`, `benchmark` e `visualize`.

- `install` executa `uv sync`;
- `lint` contém os comandos e flags exigidos pelo subject;
- `clean` remove caches e bytecode;
- `test` executa o Pytest;
- `model-check` aciona uma inspeção manual da API do SDK.

`run` e `debug` executam agora o pipeline real. `model-check` e a opção
`--validate-only` preservam caminhos rápidos para inspecionar o SDK ou validar as
entradas sem iniciar uma geração completa.

### 3.3 Modelos de dados

`src/models.py` contém modelos Pydantic para:

- `TypeDefinition`: tipo JSON declarado;
- `FunctionDefinition`: nome, descrição, parâmetros e retorno de uma função;
- `PromptInput`: um pedido natural não vazio;
- `FunctionCallResult`: as três chaves do objeto final.

Os modelos de função e prompt rejeitam campos inesperados, e os textos essenciais
têm tamanho mínimo. O modelo de resultado também rejeita chaves extras. Essa é
uma boa base para validar as fronteiras do programa.

`FunctionCallResult.parameters` usa agora valores escalares tipados e só pode ser
construído com o catálogo de funções no contexto de validação. O builder público
rejeita nome desconhecido, parâmetros ausentes ou extras, tipos incorretos,
booleanos usados como números e floats não finitos.

`FunctionCatalog` exige ao menos uma função e nomes únicos. O contrato obrigatório
suporta `string`, `number`, `integer` e `boolean`. `array` e `object` são rejeitados
explicitamente na entrada: estruturas complexas aparecem como bônus no subject e
não devem ser anunciadas sem uma gramática que realmente consiga garanti-las.

### 3.4 Leitura e escrita JSON

`src/io.py` já oferece:

- leitura com context manager;
- mensagens próprias para arquivo ausente, JSON malformado e falha de I/O;
- validação Pydantic das listas de funções e prompts;
- serialização de resultados validados;
- criação automática do diretório pai da saída;
- JSON UTF-8, sem ASCII forçado, indentado e terminado por nova linha.

Isso atende a robustez de I/O pedida. A exceção `InputFileError` unifica as falhas
de entrada e saída e inclui casos de UTF-8 inválido.

### 3.5 Interface de linha de comando

`src/__main__.py` já implementa os três argumentos do subject:

- `--functions_definition`, padrão
  `data/input/functions_definition.json`;
- `--input`, padrão `data/input/function_calling_tests.json`;
- `--output`, padrão `data/output/function_calling_results.json`.

Há ainda três opções de desenvolvimento:

- `--validate-only`, que valida entradas sem carregar o modelo;
- `--inspect-model`, que demonstra encode, logits e caminho do vocabulário.
- `--visualize`, que grava opcionalmente um relatório HTML do decoding.

O fluxo atual valida as entradas, carrega o cliente quando necessário e traduz
erros esperados em mensagens no `stderr` e códigos de saída diferentes de zero.
Sem opções auxiliares, o comando agora carrega um decoder, processa todos os
prompts em ordem e grava o array no caminho `--output`. O modelo e o vocabulário
são carregados uma vez por lote. Um lote vazio grava `[]` sem carregar o modelo.

### 3.6 Integração inicial com o modelo

`src/model.py` cria `QwenClient`, um adaptador de carregamento tardio para
`Qwen/Qwen3-0.6B`. Ele usa somente os métodos públicos relevantes do SDK:

- `encode`;
- `get_logits_from_input_ids`;
- `get_path_to_vocab_file`.

O carregamento tardio permite validar arquivos sem carregar ou baixar o modelo,
e falhas do SDK são convertidas em `ModelLoadError`. Não foi encontrado acesso a
atributos ou métodos privados do SDK no código de `src/`.

O próprio SDK fornecido depende de PyTorch, Transformers e Hugging Face. Essa é a
implementação interna do pacote entregue; o código do projeto não importa esses
pacotes diretamente. A proibição do subject deve continuar sendo respeitada no
código autoral: não se deve contornar a API pública do SDK nem implementar a
solução diretamente com Transformers.

`QwenClient` foi migrado de `dataclass` comum para `BaseModel` Pydantic. A instância
do SDK permanece em atributo privado Pydantic e continua sendo carregada somente
quando necessária.

### 3.7 Dados de demonstração e testes

Os arquivos em `data/input/` oferecem duas funções (`fn_add_numbers` e
`fn_greet`) e dois prompts correspondentes. São dados demonstrativos válidos e
não estão codificados no código-fonte.

Existem 58 testes cobrindo os dados demonstrativos, JSON inválido, catálogo
duplicado, validação dinâmica, tipos escalares, números não finitos e a gramática
incremental. A gramática é testada com funções alternativas, fragmentos, escapes,
Unicode, números, função sem argumentos, vocabulário byte-level, máscara de
logits, limite de geração, violações de estrutura/schema, escrita atômica, CLI,
cache e visualização. A integração também foi exercitada com o modelo real.

### 3.8 Gramática incremental

`src/grammar.py` implementa uma gramática Pydantic imutável para o JSON canônico:

```json
{"name":"nome_declarado","parameters":{"argumento":"valor"}}
```

Ela mantém todas as funções possíveis enquanto o prefixo ainda é ambíguo, aceita
fragmentos inteiros por `can_accept`/`advance` e só termina quando nome, chaves,
ordem, tipos e fechamento coincidem com uma definição. Strings incluem escapes e
Unicode; números seguem a sintaxe JSON; booleanos são `true`/`false`. Conteúdo
extra após o documento é rejeitado. O `prompt` conhecido fica fora da inferência e
será anexado pelo builder validado do resultado.

### 3.9 Documentação

`README.md` está escrito em inglês e inclui a primeira linha curricular,
descrição, instruções, recursos, uso de IA, algoritmo, decisões, desempenho,
desafios, testes e exemplos. `Log_de_desenvolvimento.md` registra o histórico
técnico e as justificativas em português.

## 4. Estado real de funcionamento

O que funciona hoje, por inspeção do código:

- parsing dos argumentos;
- leitura e validação básica dos dois arquivos de entrada;
- escrita de uma lista de resultados já construída por outro componente;
- adaptação inicial das operações públicas do SDK;
- catálogo e validação dinâmica exata das chamadas;
- gramática incremental de JSON para tipos escalares;
- carregamento e decodificação do vocabulário byte-level público;
- filtro de tokens inteiros pela gramática;
- prompt de seleção e extração;
- máscara explícita de logits inválidos com `-inf`;
- geração greedy token a token, limite seguro e validação final;
- modo de validação sem modelo;
- modo de inspeção do modelo, desde que dependências e pesos estejam disponíveis;
- processamento de todos os prompts em uma única carga do modelo;
- escrita atômica do array final no caminho solicitado.
- visualização HTML opcional das decisões token a token.

O que permanece como limitação:

- medição de acurácia estatisticamente ampla ou com o conjunto privado;
- tempo abaixo de cinco minutos no benchmark diagnóstico ampliado de quatro casos;
- suporte a argumentos complexos, que pertence ao bônus.

Portanto, o projeto possui agora um **pipeline funcional de ponta a ponta** para
os tipos escalares suportados, validado com o Qwen real. Restam somente validações
que dependem do ambiente externo: Moulinette privada e desempenho no hardware da
avaliação.

## 5. Conformidade e pendências

| Requisito do subject | Estado | Observação |
|---|---:|---|
| Python 3.10+ e `uv` | Conforme | Python `>=3.10`; execução auditada em 3.14.4 |
| Dependências exigidas | Conforme | `uv sync --locked` passou; SDK local, NumPy e Pydantic |
| CLI e caminhos padrão | Conforme | Pipeline real usa padrões e caminhos personalizados |
| Leitura robusta de JSON | Conforme | Ausente, malformado, UTF-8 inválido e schema inválido testados |
| Classes com Pydantic | Conforme no código autoral | Estado/configuração usam modelos Pydantic |
| Type hints e docstrings | Conforme | `mypy --strict` e `flake8` passaram |
| API pública de `llm_sdk` | Conforme no código atual | Nenhum acesso privado em `src/` |
| Escolha pelo LLM | Implementada por chamada | Maior logit válido; sem heurística textual |
| Constrained decoding | Implementado por chamada | Vocabulário, gramática e máscara `-inf` integrados |
| JSON/schema 100% válidos | Verificado nos testes atuais | Gramática, validação final e escrita atômica |
| Saída com chaves exatas | Conforme nos testes atuais | Pipeline gera somente as três chaves |
| Desempenho e acurácia | Medidos, com limite conhecido | 4/4 corretos; 348,502s no diagnóstico ampliado |
| Makefile obrigatório | Conforme | `run` executa pipeline real; alvos obrigatórios presentes |
| Testes | Boa cobertura atual | 58 testes, benchmark e execução real ponta a ponta |
| `.gitignore` | Conforme | Inclui artefatos Python e `data/output/` |
| README obrigatório em inglês | Conforme | Todas as seções exigidas estão presentes |

Verificação acumulada: 58 testes passaram; `flake8 .`, o `mypy` obrigatório e
`mypy --strict` passaram; a API real retornou 151.936 logits e 150.195 tokens
UTF-8 utilizáveis. Em 24/08/2026, a CLI padrão processou os dois prompts
corretamente com `Qwen/Qwen3-0.6B` em 148,89s. O benchmark diagnóstico anterior
obteve 100% de validade e acurácia em quatro casos, mas levou 348,502s.

## 6. Próximos passos recomendados

### Etapas 1 a 9 — Implementadas

Contratos, gramática, vocabulário, máscara, geração, pipeline, escrita atômica,
58 testes, benchmark reproduzível, README e bônus de visualização/cache foram
implementados. O próximo foco não é uma nova etapa estrutural, mas a validação
externa com o conjunto privado/moulinette e desempenho em hardware de avaliação.

## 7. Ordem prática de implementação

A sequência registrada ficou:

1. ~~schema dinâmico e testes~~ — concluído;
2. ~~gramática/parser incremental e testes~~ — concluído;
3. ~~adaptação do vocabulário e testes de tokens~~ — concluído;
4. ~~loop de constrained decoding~~ — concluído;
5. ~~escolha e argumentos pelo modelo~~ — concluído;
6. ~~pipeline completo de arquivos~~ — concluído;
7. ~~casos extremos e medição local~~ — concluídos; desempenho ampliado pendente;
8. ~~lint, tipagem e README final~~ — concluídos;
9. ~~bônus isolados e demonstráveis~~ — visualização, cache observável, suíte e
   recuperação atômica implementados; bônus não implementados não são alegados.

Essa ordem mantém o foco no requisito avaliativo central: o modelo decide o
conteúdo, enquanto o decoder garante estruturalmente que nenhuma saída inválida
possa ser produzida.

## 8. Auditoria final contra subject e régua de avaliação

### Evidências reproduzidas nesta auditoria

- `uv sync --locked`: 77 pacotes resolvidos e 70 conferidos, sem erro;
- `make install`, `make test`, `make lint` e `make lint-strict`: passaram;
- 58 testes: todos passaram;
- `make run` com Qwen real e dados padrão: código 0 em 148,89s;
- pico de memória observado na rota padrão: 5.237.620 KiB;
- saída relida com `jq`: array válido, dois resultados corretos e exatamente as
  chaves `prompt`, `name` e `parameters`;
- arquivo ausente, JSON malformado e UTF-8 inválido: código 1 e mensagem clara,
  sem traceback;
- `make visualize` com Qwen real: código 0 em 145,26s, dois traces HTML, 31
  decisões de token e 5 hits de cache observados no segundo prompt;
- nenhum import direto das bibliotecas proibidas no código autoral;
- integração autoral com o SDK limitada a `encode`,
  `get_logits_from_input_ids` e `get_path_to_vocab_file`.

### Comparação por bloco de avaliação

| Bloco | Parecer | Evidência ou ressalva |
|---|---:|---|
| Estrutura e dependências | Passa localmente | `src/`, SDK, dados, `pyproject.toml`, lock e `uv sync` válidos |
| Entrada | Passa localmente | Leitura tipada e erros controlados, inclusive UTF-8 inválido |
| Constrained decoding | Passa | Logits reais, máscara `-inf`, gramática incremental e seleção greedy |
| Escolha pelo LLM | Passa | Não há seletor por palavras-chave ou respostas codificadas |
| JSON e schema | Passa nos casos exercitados | Gramática mais três validações finais; chaves exatas do subject |
| API do SDK | Passa por inspeção | Somente operações públicas chamadas pelo adaptador |
| Makefile | Passa | Todos os alvos obrigatórios existem e os não interativos foram executados |
| Qualidade | Passa | 58 testes, `flake8`, `mypy` obrigatório e `mypy --strict` |
| README | Passa | Primeira linha e todas as seções obrigatórias em inglês |
| Acurácia | Evidência positiva, não garantia | 2/2 padrão e 4/4 diagnóstico; conjunto ainda pequeno |
| Desempenho | Passa no lote padrão, risco no privado | 148,89s padrão; benchmark de quatro casos marcou 348,502s |
| Moulinette | Não verificável neste repositório | O anexo privado e seu README não estão disponíveis |
| Bônus | Passa nos bônus alegados | Visualização real, cache observável, recuperação atômica e suíte ampla |

### Divergências e riscos para a defesa

1. O subject v1.5 exige `prompt`, `name` e `parameters`; a régua complementar
   capturada menciona `prompt`, `fn_name` e `args`. O projeto segue o subject,
   que é a fonte primária e contém definição e exemplo coerentes entre si. Essa
   divergência deve ser mostrada ao avaliador antes de alterar qualquer chave.
2. A régua fala em valores restritos a opções, mas o formato de entrada do subject
   só define `type`. Campos de enumeração não são aceitos sem contrato oficial,
   para não fingir uma capacidade não especificada.
3. Um prompt sem função correspondente não possui representação de “nenhuma
   chamada” no formato obrigatório. O programa continua escolhendo a alternativa
   de maior logit válida; isso é uma limitação semântica, não uma quebra de JSON.
4. O tempo cresce aproximadamente com a quantidade e o comprimento dos prompts.
   A execução padrão passa, porém o benchmark ampliado ultrapassou cinco minutos;
   um conjunto privado maior pode falhar o requisito de desempenho nesta máquina.
5. A Moulinette privada é o único teste que pode confirmar o score exigido pela
   régua. Não é correto prometer aprovação sem executá-la.
6. No estado atual do working tree, muitos arquivos da implementação ainda estão
   como não rastreados ou modificados. A 42 avalia somente o que estiver no Git:
   é obrigatório revisar, adicionar, commitar e enviar esses arquivos antes da
   defesa. O diretório `data/output/` deve permanecer ignorado.

### Veredito

Tecnicamente, a parte obrigatória está implementada e a defesa manual tem boa
probabilidade de aprovação: instalação, execução padrão, constrained decoding,
JSON/schema, qualidade e documentação passaram localmente. Ainda assim, o projeto
**não pode ser declarado garantidamente aprovado** enquanto os arquivos não forem
commitados/enviados e a Moulinette privada não for executada. O maior risco
funcional restante é o tempo para um lote privado maior; o maior risco imediato
de entrega é o estado não rastreado do Git.
