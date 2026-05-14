# ============================================================
# tasks.py
# Define as tarefas atribuídas a cada agente do sistema
# Uma Task descreve: O QUE fazer, QUAL resultado entregar, e QUEM executa
#
# COMPATÍVEL COM: CrewAI >= 0.80 | Python 3.11+
#
# NOTA sobre `context`:
#   No CrewAI >= 0.80, Task aceita o parâmetro `context` como
#   uma lista de Tasks cujo output será injetado como contexto.
#   Isso funciona da mesma forma que antes, mas os agentes são
#   passados como argumento (agent=...) — NÃO instanciados aqui.
#   Os imports de agentes foram removidos deste arquivo para
#   evitar inicialização duplicada do LLM.
# ============================================================

from crewai import Task


# ==============================================================
# TAREFA 1: PESQUISA DE MERCADO
# Executada pelo: Analista de Mercado (News Scout)
# ==============================================================
def create_market_research_task(agent):
    """
    Tarefa de coleta e análise de dados de mercado.

    O agente deve buscar preços atuais e tendências para
    MAP e Ureia, tanto no mercado internacional quanto no Brasil.

    Args:
        agent: O agente Analista de Mercado que vai executar esta tarefa
    """
    return Task(
        description=(
            """
            Realize uma análise completa do mercado de MAP (Monoamônio Fosfato, 11-52-00)
            e Ureia (46% N) para o mercado brasileiro. Siga estes passos:

            1. PREÇOS INTERNACIONAIS (FOB):
               - Busque o preço atual de MAP FOB Tampa (EUA) em USD/tonelada
               - Busque o preço atual de Ureia FOB Yuzhne (Ucrânia/Mar Negro) em USD/tonelada
               - Verifique preços alternativos: MAP FOB China, Ureia FOB Oriente Médio

            2. PREÇOS BRASIL (CFR/Spot):
               - Busque o preço CFR Santos/Paranaguá para MAP em USD/tonelada
               - Busque o preço spot no mercado interno brasileiro para MAP em R$/tonelada
               - Faça o mesmo para Ureia
               - Considere que o câmbio USD/BRL impacta o preço em reais

            3. CONTEXTO DE MERCADO:
               - Identifique a fase atual do ciclo agrícola brasileiro
                 (pré-safra soja: Ago-Out / pré-safra milho: Nov-Jan são épocas de alta demanda)
               - Busque notícias recentes sobre disponibilidade e problemas logísticos
               - Verifique se há filas ou congestionamentos nos portos de Santos e Paranaguá

            4. FATORES DE RISCO:
               - Políticas de exportação da China (principal fornecedor de MAP)
               - Situação geopolítica que afete fornecedores russos de Ureia
               - Variação do câmbio USD/BRL nos últimos 30 dias

            Use a ferramenta de busca para encontrar dados reais e atualizados.
            Cite as fontes quando possível (Argus, Green Markets, Reuters, Bloomberg Agri).
            """
        ),
        expected_output=(
            """
            Um relatório estruturado em Português contendo:

            ## RELATÓRIO DE MERCADO - MAP e UREIA
            **Data da análise:** [data atual]

            ### 1. Preços Internacionais (FOB)
            - MAP FOB Tampa: $XXX/t
            - MAP FOB China: $XXX/t
            - Ureia FOB Yuzhne: $XXX/t
            - Ureia FOB Oriente Médio: $XXX/t

            ### 2. Preços Brasil
            - MAP CFR Santos: $XXX/t
            - MAP Spot interno: R$ XXX/t
            - Ureia CFR Santos: $XXX/t
            - Ureia Spot interno: R$ XXX/t
            - Câmbio USD/BRL: X.XX

            ### 3. Contexto de Mercado
            [Análise da situação atual, sazonalidade, demanda]

            ### 4. Fatores de Risco
            [Lista dos principais riscos identificados]

            ### 5. Conclusão do Analista
            [Avaliação geral: mercado favorável ou desfavorável para arbitragem?]
            """
        ),
        agent=agent,
    )


# ==============================================================
# TAREFA 2: CÁLCULO DE ARBITRAGEM
# Executada pelo: Estrategista de Arbitragem (Math Genius)
# ==============================================================
def create_arbitrage_task(agent, context_tasks):
    """
    Tarefa de cálculo de paridade e identificação de janelas de arbitragem.

    Usa os dados coletados na Tarefa 1 como insumo (context).
    Realiza os cálculos financeiros para determinar se a operação é lucrativa.

    Args:
        agent: O agente Estrategista de Arbitragem
        context_tasks: Lista com a tarefa de pesquisa (para usar como contexto)
    """
    return Task(
        description=(
            """
            Com base no relatório de mercado fornecido, calcule a paridade de importação
            e identifique janelas de arbitragem para MAP e Ureia. Execute os cálculos abaixo:

            ## ESTRUTURA DE CUSTOS PARA CÁLCULO (use valores do relatório de mercado):

            ### FÓRMULA DE PARIDADE DE IMPORTAÇÃO:
            Custo CIF Brasil = Preço FOB + Frete Marítimo + Seguro

            Custo Total Desembaraçado = Custo CIF
                                       + Tarifa de Importação (0% - fertilizantes são isentos no Brasil)
                                       + AFRMM - Adicional de Frete (25% do frete marítimo, apenas em navios nacionais; use 0% se navio estrangeiro)
                                       + THC - Terminal Handling Charge (~$15-20/t)
                                       + Custo de Armazenagem Porto (~$8-12/t)
                                       + Frete Interno (porto → distribuidor, estimativa: $20-40/t dependendo do estado)
                                       + Margem do Broker (alvo: $10-15/t ou 2-3% sobre CIF)

            ### PARÂMETROS LOGÍSTICOS PARA USAR:
            - Frete marítimo Tampa→Santos (MAP): pesquise valor atual, estimativa $35-55/t
            - Frete marítimo Yuzhne→Santos (Ureia): pesquise valor atual, estimativa $30-45/t
            - Seguro: 0.3% sobre (FOB + Frete)
            - Tempo de trânsito Tampa→Santos: ~18-22 dias
            - Tempo de trânsito Yuzhne→Santos: ~25-30 dias
            - Fila estimada em Santos/Paranaguá: busque informação atual (pode ser 5-15 dias)

            ### CÁLCULOS NECESSÁRIOS:
            1. Calcule o Custo Total de Importação para MAP (em USD/t e em R$/t)
            2. Calcule o Custo Total de Importação para Ureia (em USD/t e em R$/t)
            3. Compare com o preço spot interno brasileiro
            4. Calcule a Janela de Arbitragem:
               Janela = Preço Spot Brasil - Custo Total Importação
               (Positivo = oportunidade lucrativa; Negativo = importação não compensa)
            5. Calcule o ROI estimado da operação (considere lote mínimo de 5.000 toneladas)
            6. Calcule o Break-even: qual o preço FOB máximo que ainda deixa lucro?

            ### ANÁLISE DE SENSIBILIDADE:
            - O que acontece se o câmbio USD/BRL variar ±5%?
            - O que acontece se o frete marítimo variar ±$10/t?
            - Qual é o prazo máximo de entrega para ainda capturar a janela de arbitragem?
            """
        ),
        expected_output=(
            """
            Um relatório financeiro detalhado contendo:

            ## ANÁLISE DE ARBITRAGEM - FERTILIZANTES
            **Referência:** [data]

            ### 1. Cálculo de Paridade - MAP
            | Item                          | USD/t    | R$/t      |
            |-------------------------------|----------|-----------|
            | Preço FOB Tampa               | $XXX     | R$ XXX    |
            | (+) Frete Marítimo            | $XX      | R$ XX     |
            | (+) Seguro (0.3%)             | $X       | R$ X      |
            | (=) Custo CIF Santos          | $XXX     | R$ XXX    |
            | (+) THC                       | $XX      | R$ XX     |
            | (+) Armazenagem Porto         | $X       | R$ X      |
            | (+) Frete Interno (estimado)  | $XX      | R$ XX     |
            | (+) Margem do Broker (alvo)   | $XX      | R$ XX     |
            | (=) CUSTO TOTAL               | $XXX     | R$ XXX    |
            | Preço Spot Mercado Brasil     | $XXX     | R$ XXX    |
            | **JANELA DE ARBITRAGEM**      | **$XX**  | **R$ XX** |

            ### 2. Cálculo de Paridade - Ureia
            [Mesma estrutura acima para Ureia]

            ### 3. Análise de Viabilidade
            - Lote mínimo considerado: 5.000 toneladas
            - Lucro bruto estimado (MAP): R$ X.XXX.XXX
            - Lucro bruto estimado (Ureia): R$ X.XXX.XXX
            - ROI estimado: XX%
            - Prazo de retorno: X meses

            ### 4. Análise de Sensibilidade
            [Tabela com cenários de câmbio e frete]

            ### 5. RECOMENDAÇÃO DO ESTRATEGISTA
            [COMPRAR / AGUARDAR / NÃO OPERAR] + justificativa técnica
            """
        ),
        agent=agent,
        context=context_tasks,             # Usa o output da tarefa de pesquisa como input
    )


# ==============================================================
# TAREFA 3: REDAÇÃO DE COMUNICAÇÕES COMERCIAIS
# Executada pelo: Agente de Comunicação (Ghostwriter)
# ==============================================================
def create_communication_task(agent, context_tasks):
    """
    Tarefa de redação de comunicações profissionais.

    Usa os resultados das Tarefas 1 e 2 para redigir comunicações
    contextualizadas com dados reais de mercado e arbitragem.

    Args:
        agent: O agente de Comunicação
        context_tasks: Lista com as tarefas anteriores (contexto)
    """
    return Task(
        description=(
            """
            Com base na análise de mercado e nos cálculos de arbitragem fornecidos,
            redija as seguintes comunicações profissionais:

            ## DOCUMENTO 1 — E-MAIL PARA FORNECEDOR INTERNACIONAL (em Inglês)
            Destinatário fictício: Mr. James Richardson, Senior Trader da empresa
            "Atlantic Phosphate Trading Ltd." (exportador de MAP baseado em Tampa, Flórida).

            O e-mail deve:
            - Apresentar sua empresa de corretagem baseada em Londres de forma profissional
            - Demonstrar conhecimento do mercado (use os dados de preço do relatório)
            - Solicitar cotação para um lote de 5.000 a 10.000 toneladas de MAP (11-52-00)
            - Especificar: origem Tampa, destino Santos (Brazil), Incoterm CFR
            - Mencionar que você tem buyer confirmado no Brasil (dá credibilidade)
            - Tom: profissional, direto, confiante — demonstre que você conhece o mercado
            - Tamanho: 150-200 palavras, formato de e-mail completo (assunto, saudação, corpo, fechamento)

            ## DOCUMENTO 2 — PROPOSTA COMERCIAL PARA COMPRADOR BRASILEIRO (em Português)
            Destinatário fictício: Sr. Roberto Carvalho, Diretor Comercial da
            "AgroSul Distribuidora" (distribuidor regional no Paraná).

            A proposta deve:
            - Apresentar a oportunidade de compra de MAP com preço competitivo
            - Incluir os dados de custo CIF calculados (use os números do cálculo de arbitragem)
            - Destacar a vantagem de preço vs. mercado interno atual
            - Especificar condições: quantidade (5.000t mínimo), pagamento (sugerir L/C ou 30% antecipado + 70% contra BL), prazo de entrega estimado
            - Incluir uma seção de riscos e como serão mitigados
            - Tom: consultivo e parceiro, não apenas vendedor — você está ajudando o cliente a economizar
            - Tamanho: 300-400 palavras, formato de proposta profissional

            Certifique-se de que os dados de preço usados nas comunicações são CONSISTENTES
            com os valores calculados na análise de arbitragem.
            """
        ),
        expected_output=(
            """
            Dois documentos profissionais completos:

            ========================================
            DOCUMENTO 1 — EMAIL TO SUPPLIER (English)
            ========================================

            **Subject:** MAP (11-52-00) Inquiry – CFR Santos, Brazil – [Quantity] MT

            Dear Mr. Richardson,

            [Corpo do e-mail em Inglês — 150-200 palavras]

            Best regards,
            [Seu nome]
            [Sua empresa]
            London, UK

            ========================================
            DOCUMENTO 2 — PROPOSTA PARA COMPRADOR (Português)
            ========================================

            **PROPOSTA COMERCIAL CONFIDENCIAL**
            **Ref:** [Número de referência]
            **Data:** [Data]
            **Para:** Sr. Roberto Carvalho — AgroSul Distribuidora

            [Proposta completa em Português — 300-400 palavras]

            Atenciosamente,
            [Seu nome]
            [Sua empresa]
            Londres, Reino Unido
            """
        ),
        agent=agent,
        context=context_tasks,             # Usa outputs das duas tarefas anteriores
    )