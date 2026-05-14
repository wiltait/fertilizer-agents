# ============================================================
# main.py
# Ponto de entrada do sistema de multi-agentes de corretagem
# de fertilizantes. Execute com: python main.py
#
# COMPATÍVEL COM: CrewAI >= 0.80 | Python 3.11+
# ============================================================

import os
from datetime import datetime
from dotenv import load_dotenv
from crewai import Crew, Process

# Importa as funções que criam os agentes
from agents import (
    create_market_analyst,
    create_arbitrage_strategist,
    create_communication_agent,
)

# Importa as funções que criam as tarefas
from tasks import (
    create_market_research_task,
    create_arbitrage_task,
    create_communication_task,
)


# ==============================================================
# PASSO 1: Carregar variáveis de ambiente do arquivo .env
# ==============================================================
load_dotenv()  # Lê o arquivo .env e disponibiliza as variáveis

# Validação: verifica se a chave da API foi carregada corretamente
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or api_key == "sua_chave_aqui":
    raise ValueError(
        "\n❌ ERRO: GOOGLE_API_KEY não configurada!\n"
        "Abra o arquivo .env e adicione sua chave real do Google AI Studio.\n"
        "Obtenha em: https://aistudio.google.com/app/apikey\n"
    )

# IMPORTANTE: O CrewAI usa LiteLLM internamente para chamar o Gemini.
# O LiteLLM espera a chave em GEMINI_API_KEY (além de GOOGLE_API_KEY).
# Esta linha garante que ambas as variáveis estão disponíveis.
os.environ["GEMINI_API_KEY"] = api_key

print("=" * 60)
print("🌱 SISTEMA DE ARBITRAGEM DE FERTILIZANTES")
print("   MAP & Ureia | London ↔ Brasil")
print("=" * 60)
print(f"✅ API Key carregada: {api_key[:8]}...{api_key[-4:]}")
print()


# ==============================================================
# PASSO 2: Instanciar os Agentes
# Cada agente é criado via função importada do agents.py
# ==============================================================
print("🤖 Inicializando agentes...")

market_analyst = create_market_analyst()
arbitrage_strategist = create_arbitrage_strategist()
communication_agent = create_communication_agent()

print("   ✅ Analista de Mercado (News Scout) — pronto")
print("   ✅ Estrategista de Arbitragem (Math Genius) — pronto")
print("   ✅ Agente de Comunicação (Ghostwriter) — pronto")
print()


# ==============================================================
# PASSO 3: Criar as Tarefas
# As tarefas são criadas e ligadas aos seus respectivos agentes.
# Note que as tarefas 2 e 3 recebem a tarefa anterior como
# 'context' — assim o CrewAI passa o output entre elas.
# ==============================================================
print("📋 Configurando tarefas...")

# Tarefa 1: Pesquisa de mercado (sem dependências externas)
task_research = create_market_research_task(agent=market_analyst)

# Tarefa 2: Cálculo de arbitragem (depende da pesquisa de mercado)
task_arbitrage = create_arbitrage_task(
    agent=arbitrage_strategist,
    context_tasks=[task_research],         # Recebe o output da Tarefa 1
)

# Tarefa 3: Comunicações comerciais (depende das duas anteriores)
task_communication = create_communication_task(
    agent=communication_agent,
    context_tasks=[task_research, task_arbitrage],  # Recebe outputs das Tarefas 1 e 2
)

print("   ✅ Tarefa 1: Pesquisa de Mercado")
print("   ✅ Tarefa 2: Cálculo de Arbitragem")
print("   ✅ Tarefa 3: Comunicações Comerciais")
print()


# ==============================================================
# PASSO 4: Montar e Executar o Crew
# O Crew orquestra os agentes e tarefas.
# Process.sequential = as tarefas rodam em ordem (1 → 2 → 3)
# ==============================================================
print("🚀 Montando o Crew...")

crew = Crew(
    agents=[
        market_analyst,
        arbitrage_strategist,
        communication_agent,
    ],
    tasks=[
        task_research,
        task_arbitrage,
        task_communication,
    ],
    process=Process.sequential,            # Execução sequencial: uma tarefa por vez, em ordem
    verbose=True,                          # Mostra detalhes da execução no terminal
    # memory=True,                         # Descomente para habilitar memória entre tarefas (requer OpenAI ou config extra)
    # max_rpm=10,                          # Descomente para limitar requests por minuto (evita rate limiting)
)

print("✅ Crew configurado!")
print()
print("=" * 60)
print("⚡ INICIANDO ANÁLISE — Isso pode levar 2-5 minutos...")
print("   Os agentes vão pensar em voz alta (verbose=True)")
print("=" * 60)
print()


# ==============================================================
# PASSO 5: Kickoff — Dispara a execução do Crew
# ==============================================================
try:
    resultado = crew.kickoff()

    # ===========================================================
    # PASSO 6: Exibir e salvar os resultados
    #
    # NOTA: No CrewAI >= 0.80, crew.kickoff() retorna um objeto
    # CrewOutput, não uma string. Usamos .raw para obter o texto
    # final, e .tasks_output para ver o output de cada tarefa.
    # ===========================================================
    print()
    print("=" * 60)
    print("✅ ANÁLISE CONCLUÍDA!")
    print("=" * 60)

    # Exibe o output de cada tarefa individualmente
    print()
    print("📋 OUTPUT POR TAREFA:")
    print()
    for i, task_output in enumerate(resultado.tasks_output, 1):
        print(f"--- Tarefa {i}: {task_output.description[:60]}... ---")
        print(task_output.raw)
        print()

    # Exibe o resultado final consolidado (output da última tarefa)
    print("=" * 60)
    print("📊 RESULTADO FINAL (Comunicações Comerciais):")
    print("-" * 60)
    print(resultado.raw)

    # Salva o relatório completo em arquivo com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"resultado_analise_{timestamp}.txt"

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("ANÁLISE DE ARBITRAGEM DE FERTILIZANTES\n")
        f.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        # Salva cada tarefa separadamente no arquivo
        for i, task_output in enumerate(resultado.tasks_output, 1):
            f.write(f"\n{'=' * 60}\n")
            f.write(f"TAREFA {i}\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(task_output.raw)
            f.write("\n")

    print()
    print(f"💾 Relatório completo salvo em: {output_filename}")
    print()

except Exception as e:
    # Tratamento de erros com mensagens úteis para debugging
    print()
    print(f"❌ ERRO durante a execução: {type(e).__name__}")
    print(f"   Detalhes: {str(e)}")
    print()
    print("💡 DICAS PARA RESOLVER:")
    print("   • Erro de modelo: Confirme 'gemini/gemini-1.5-flash' em agents.py")
    print("   • Erro de autenticação: Verifique GOOGLE_API_KEY no arquivo .env")
    print("   • Erro de rate limit: Aguarde 1 minuto e tente novamente")
    print("   • Erro de importação: Rode 'pip install -r requirements.txt'")
    print("   • Ainda com problemas? Tente: pip install --upgrade crewai crewai-tools")
    raise  # Re-lança o erro para ver o traceback completo


# ==============================================================
# COMO EXECUTAR:
#   1. Configure o .env com sua GOOGLE_API_KEY
#   2. pip install -r requirements.txt
#   3. python main.py
#
# ESTRUTURA DE ARQUIVOS ESPERADA:
#   fertilizer_broker/
#   ├── .env                        ← Sua chave da API (nunca suba para o Git!)
#   ├── main.py                     ← Este arquivo
#   ├── agents.py                   ← Definição dos 3 agentes
#   ├── tasks.py                    ← Definição das 3 tarefas
#   ├── requirements.txt            ← Dependências do projeto
#   └── resultado_analise_TIMESTAMP.txt  ← Gerado após cada execução
# ==============================================================