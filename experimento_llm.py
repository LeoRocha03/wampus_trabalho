#!/usr/bin/env python3
"""Bônus: Agente D (LLM) × Agente B (lógico) nas MESMAS sementes.

  python experimento_llm.py                 # 30 mundos (sementes 0..29)
  python experimento_llm.py -n 50 --log chamadas.jsonl

Precisa de uma chave de API (XAI_API_KEY, DASHSCOPE_API_KEY ou ANTHROPIC_API_KEY)
no ambiente ou no arquivo .env. Veja agentes/agente_llm.py.
"""

import argparse
import csv
import os
import statistics
import sys

from wumpus import rodar_episodio
from agentes.agente_logico import AgenteLogico
from agentes.agente_llm import AgenteLLM, PROVEDORES, provedor_configurado


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=30)
    p.add_argument("--semente-base", type=int, default=0)
    p.add_argument("--log", help="grava prompts e respostas do LLM (jsonl)")
    p.add_argument("--csv", default="resultados_llm.csv")
    a = p.parse_args()
    if a.log:
        os.environ["WUMPUS_LLM_LOG"] = a.log

    provedor = provedor_configurado()
    modelo = AgenteLLM().modelo
    print(f"Provedor: {provedor} · modelo: {modelo} · {a.n} mundos\n", file=sys.stderr)

    linhas = []
    for i, s in enumerate(range(a.semente_base, a.semente_base + a.n), 1):
        rb = rodar_episodio(AgenteLogico, s)
        caixa = []

        def fabrica():
            caixa.append(AgenteLLM())
            return caixa[-1]

        rd = rodar_episodio(fabrica, s)
        d = caixa[-1]
        if rd.erro:
            print(f"\n[!] erro na semente {s}:\n{rd.erro}", file=sys.stderr)
            if i == 1:
                return 1
        linhas.append({
            "semente": s, "pont_B": rb.pontuacao, "desfecho_B": rb.desfecho, "acoes_B": rb.acoes,
            "tempo_B_ms": rb.tempo_s * 1000,
            "pont_D": rd.pontuacao, "desfecho_D": rd.desfecho, "acoes_D": rd.acoes,
            "tempo_D_s": rd.tempo_s, "chamadas": d.chamadas,
            "tokens_in": d.tokens_entrada, "tokens_out": d.tokens_saida,
            "movimentos": d.movimentos, "inseguros": d.movimentos_inseguros,
            "afirmacoes": d.afirmacoes, "alucinacoes": d.alucinacoes, "sem_prova": d.sem_prova,
            "invalidas": d.respostas_invalidas,
            "exemplo_alucinacao": " | ".join(d.exemplos_alucinacao),
        })
        print(f"\r  {i}/{a.n} mundos", end="", file=sys.stderr)
    print(file=sys.stderr)

    with open(a.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)

    def media(k):
        return statistics.mean(l[k] for l in linhas)

    def dp(k):
        return statistics.stdev(l[k] for l in linhas) if len(linhas) > 1 else 0.0

    def taxa(k, v):
        return 100 * sum(l[k] in v for l in linhas) / len(linhas)

    morte = {"caiu_no_poco", "devorado_pelo_wumpus"}
    n_mov = sum(l["movimentos"] for l in linhas) or 1
    n_af = sum(l["afirmacoes"] for l in linhas) or 1
    fmt = lambda x: f"{x:.1f}".replace(".", ",")
    print(f"## D × B · {provedor} ({modelo}) · {a.n} mundos\n")
    print("| Agente | Pontuação (média ± dp) | Sucesso | Morte | Ações | Tempo/partida |")
    print("|---|---:|---:|---:|---:|---:|")
    print(f"| B · Lógico | {fmt(media('pont_B'))} ± {fmt(dp('pont_B'))} | {fmt(taxa('desfecho_B', {'saiu_com_ouro'}))}% "
          f"| {fmt(taxa('desfecho_B', morte))}% | {fmt(media('acoes_B'))} | {media('tempo_B_ms'):.1f} ms |")
    print(f"| D · LLM | {fmt(media('pont_D'))} ± {fmt(dp('pont_D'))} | {fmt(taxa('desfecho_D', {'saiu_com_ouro'}))}% "
          f"| {fmt(taxa('desfecho_D', morte))}% | {fmt(media('acoes_D'))} | {media('tempo_D_s'):.1f} s |")
    print("\n## Métricas do slide 14 (Agente D)\n")
    print(f"- Taxa de erros: {sum(l['inseguros'] for l in linhas)} de {n_mov} movimentos "
          f"({100 * sum(l['inseguros'] for l in linhas) / n_mov:.1f}%) entraram em casa que a lógica não prova segura")
    print(f"- Alucinações: {sum(l['alucinacoes'] for l in linhas)} de {n_af} afirmações "
          f"({100 * sum(l['alucinacoes'] for l in linhas) / n_af:.1f}%) contradizem o que a BC prova")
    print(f"- Afirmações 'segura' sem prova: {sum(l['sem_prova'] for l in linhas)}")
    print(f"- Respostas inválidas (JSON/ação): {sum(l['invalidas'] for l in linhas)}")
    print(f"- Custo por partida: {media('chamadas'):.1f} chamadas, "
          f"{media('tokens_in'):.0f} tokens de entrada + {media('tokens_out'):.0f} de saída, "
          f"{media('tempo_D_s'):.1f} s")
    print(f"\nDetalhe por mundo em {a.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
