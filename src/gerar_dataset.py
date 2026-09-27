import random
import re
import unicodedata
from pathlib import Path

import pandas as pd

RANDOM_STATE = 42
MIN_VARIACOES = 60
MAX_VARIACOES = 120

CATALOGO = {
    "potencia": {
        "unidades": ["cv", "hp", "ps"],
        "sementes": [
            "potencia", "potência", "potencia do motor", "potência máxima",
            "cavalos", "cavalos de potencia", "cavalos de força", "hp",
            "horsepower", "horse power", "cv", "quantos cavalos",
            "força do motor", "poder do motor", "potencia maxima do motor",
        ],
    },
    "torque": {
        "unidades": ["nm", "kgfm", "lb-ft"],
        "sementes": [
            "torque", "torque maximo", "torque máximo", "torque do motor",
            "torque max", "nm de torque", "força de torque", "torque maximo nm",
            "quanto de torque", "torque em nm", "torque em kgfm",
        ],
    },
    "cilindrada": {
        "unidades": ["l", "cm3", "litros"],
        "sementes": [
            "cilindrada", "cilindrada do motor", "tamanho do motor",
            "capacidade do motor", "motor em litros", "displacement",
            "cilindradas", "litragem do motor", "quantos litros o motor tem",
        ],
    },
    "numero_cilindros": {
        "unidades": ["cilindros"],
        "sementes": [
            "numero de cilindros", "número de cilindros", "quantos cilindros",
            "cilindros do motor", "qtd de cilindros", "configuracao de cilindros",
            "motor v6 ou v8", "numero cilindros",
        ],
    },
    "tipo_combustivel": {
        "unidades": [],
        "sementes": [
            "tipo de combustivel", "tipo de combustível", "combustivel usado",
            "gasolina ou diesel", "fuel type", "que combustivel usa",
            "abastece com o que", "diesel ou flex",
        ],
    },
    "consumo_cidade": {
        "unidades": ["km/l"],
        "sementes": [
            "consumo na cidade", "consumo urbano", "km por litro na cidade",
            "consumo cidade", "gasto de combustivel na cidade",
            "eficiencia na cidade", "quanto consome na cidade",
        ],
    },
    "consumo_estrada": {
        "unidades": ["km/l"],
        "sementes": [
            "consumo na estrada", "consumo rodoviario", "km por litro na estrada",
            "consumo estrada", "gasto de combustivel na estrada",
            "eficiencia na rodovia", "quanto consome na estrada",
        ],
    },
    "consumo_combinado": {
        "unidades": ["km/l"],
        "sementes": [
            "consumo combinado", "consumo medio", "consumo médio geral",
            "media de consumo", "km/l combinado", "consumo geral do carro",
        ],
    },
    "autonomia": {
        "unidades": ["km"],
        "sementes": [
            "autonomia", "autonomia do tanque", "alcance com tanque cheio",
            "quantos km roda com tanque cheio", "autonomia maxima",
            "distancia com um tanque",
        ],
    },
    "capacidade_tanque": {
        "unidades": ["litros", "l"],
        "sementes": [
            "capacidade do tanque", "tamanho do tanque", "tanque de combustivel",
            "quantos litros cabe no tanque", "capacidade tanque combustivel",
            "volume do tanque",
        ],
    },
    "transmissao": {
        "unidades": [],
        "sementes": [
            "tipo de transmissao", "tipo de transmissão", "cambio",
            "câmbio", "quantidade de marchas", "transmissao automatica ou manual",
            "numero de marchas", "gearbox", "cambio automatico",
        ],
    },
    "tracao": {
        "unidades": [],
        "sementes": [
            "tipo de tracao", "tipo de tração", "tracao 4x4 ou 4x2",
            "sistema de tracao", "drivetrain", "tracao integral",
            "e 4x4 ou nao",
        ],
    },
    "comprimento": {
        "unidades": ["mm", "m"],
        "sementes": [
            "comprimento", "comprimento total", "comprimento do veiculo",
            "tamanho do carro", "quao comprido e o carro", "comprimento em mm",
            "extensao do veiculo",
        ],
    },
    "largura": {
        "unidades": ["mm", "m"],
        "sementes": [
            "largura", "largura do veiculo", "largura total",
            "quao largo e o carro", "largura em mm", "largura sem espelhos",
        ],
    },
    "altura": {
        "unidades": ["mm", "m"],
        "sementes": [
            "altura", "altura do veiculo", "altura total",
            "quao alto e o carro", "altura em mm",
        ],
    },
    "entre_eixos": {
        "unidades": ["mm"],
        "sementes": [
            "entre eixos", "entre-eixos", "distancia entre eixos",
            "wheelbase", "espaco entre os eixos", "medida entre eixos",
        ],
    },
    "vao_livre_solo": {
        "unidades": ["mm"],
        "sementes": [
            "vao livre do solo", "vão livre do solo", "altura livre do solo",
            "ground clearance", "distancia ao solo", "altura em relacao ao chao",
        ],
    },
    "peso_vazio": {
        "unidades": ["kg"],
        "sementes": [
            "peso vazio", "peso em ordem de marcha", "peso do carro vazio",
            "quanto pesa o carro", "massa em ordem de marcha", "peso curb weight",
        ],
    },
    "peso_bruto": {
        "unidades": ["kg"],
        "sementes": [
            "peso bruto total", "pbt", "peso maximo permitido",
            "capacidade maxima de peso", "gross vehicle weight",
        ],
    },
    "capacidade_carga": {
        "unidades": ["kg"],
        "sementes": [
            "capacidade de carga", "quanto pode carregar", "carga util",
            "carga máxima", "payload", "capacidade de carga util",
        ],
    },
    "capacidade_cacamba": {
        "unidades": ["l", "litros", "mm"],
        "sementes": [
            "capacidade da cacamba", "capacidade da caçamba", "tamanho da cacamba",
            "tamanho da caçamba", "volume da caçamba", "espaco da cacamba",
            "dimensoes da caçamba", "quanto cabe na cacamba", "bed capacity",
        ],
    },
    "capacidade_reboque": {
        "unidades": ["kg"],
        "sementes": [
            "capacidade de reboque", "quanto pode rebocar", "peso maximo rebocavel",
            "towing capacity", "capacidade de tracionamento",
        ],
    },
    "velocidade_maxima": {
        "unidades": ["km/h"],
        "sementes": [
            "velocidade maxima", "velocidade máxima", "top speed",
            "qual a velocidade maxima", "velocidade final",
        ],
    },
    "aceleracao_0_100": {
        "unidades": ["s", "segundos"],
        "sementes": [
            "aceleracao de 0 a 100", "0 a 100 km/h", "tempo de aceleracao",
            "quanto tempo para chegar a 100", "0-100", "aceleracao zero a cem",
        ],
    },
    "diametro_rodas": {
        "unidades": ["polegadas", "aro"],
        "sementes": [
            "diametro das rodas", "aro das rodas", "tamanho do aro",
            "aro das rodas em polegadas", "medida das rodas", "wheel size",
        ],
    },
    "tipo_freio": {
        "unidades": [],
        "sementes": [
            "tipo de freio", "sistema de freios", "freio a disco ou tambor",
            "brake type", "freios dianteiros e traseiros",
        ],
    },
    "airbags": {
        "unidades": ["unidades"],
        "sementes": [
            "numero de airbags", "quantidade de airbags", "quantos airbags",
            "airbags de serie", "airbags disponiveis",
        ],
    },
    "capacidade_porta_malas": {
        "unidades": ["litros", "l"],
        "sementes": [
            "capacidade do porta malas", "capacidade do porta-malas",
            "volume do porta malas", "tamanho do porta-malas", "trunk capacity",
        ],
    },
    "numero_portas": {
        "unidades": ["portas"],
        "sementes": [
            "numero de portas", "quantidade de portas", "quantas portas tem",
            "e de 2 ou 4 portas",
        ],
    },
    "numero_lugares": {
        "unidades": ["lugares", "pessoas"],
        "sementes": [
            "numero de lugares", "quantidade de lugares", "quantos lugares",
            "capacidade de passageiros", "quantas pessoas cabem",
        ],
    },
    "ano_modelo": {
        "unidades": [],
        "sementes": [
            "ano do modelo", "ano de fabricacao", "ano modelo",
            "qual o ano do carro", "model year",
        ],
    },
    "preco": {
        "unidades": ["r$", "reais"],
        "sementes": [
            "preco", "preço", "valor do carro", "preco de tabela",
            "quanto custa", "preco sugerido", "valor de venda",
        ],
    },
    "cor": {
        "unidades": [],
        "sementes": [
            "cor disponivel", "cores disponiveis", "cor do veiculo",
            "opcoes de cor", "qual a cor",
        ],
    },
    "garantia": {
        "unidades": ["anos", "km"],
        "sementes": [
            "garantia", "tempo de garantia", "garantia de fabrica",
            "garantia do veiculo", "quanto tempo de garantia",
        ],
    },
    "tipo_suspensao": {
        "unidades": [],
        "sementes": [
            "tipo de suspensao", "tipo de suspensão", "sistema de suspensao",
            "suspensao dianteira e traseira", "suspension type",
        ],
    },
    "angulo_ataque": {
        "unidades": ["graus"],
        "sementes": [
            "angulo de ataque", "ângulo de ataque", "approach angle",
            "angulo de entrada off-road",
        ],
    },
    "angulo_saida": {
        "unidades": ["graus"],
        "sementes": [
            "angulo de saida", "ângulo de saída", "departure angle",
            "angulo de saida off-road",
        ],
    },
    "profundidade_vau": {
        "unidades": ["mm"],
        "sementes": [
            "profundidade de vau", "capacidade de vau", "wading depth",
            "profundidade maxima de agua", "quanto de agua o carro atravessa",
        ],
    },
    "sistema_infotainment": {
        "unidades": ["polegadas"],
        "sementes": [
            "sistema multimidia", "central multimidia", "tamanho da tela multimidia",
            "infotainment", "tela central do carro",
        ],
    },
    "sensor_estacionamento": {
        "unidades": [],
        "sementes": [
            "sensor de estacionamento", "camera de re", "câmera de ré",
            "sensor de re", "camera 360", "assistente de estacionamento",
        ],
    },
}

FORA_DO_CATALOGO_SEMENTES = [
    "concessionaria mais proxima", "onde comprar", "telefone da revenda",
    "agendar test drive", "horario de funcionamento da loja",
    "opinioes de donos", "avaliacao dos jornalistas", "comparativo com concorrentes",
    "financiamento disponivel", "taxa de juros do financiamento", "consorcio do carro",
    "seguro do veiculo", "valor do seguro", "ipva do carro", "tabela fipe",
    "revenda do carro usado", "manual do proprietario", "revisao programada",
    "custo da revisao", "itens de serie", "itens opcionais", "pacotes de acessorios",
    "acessorios originais", "cor da capota", "estilo do banco", "material do banco",
    "assistente de voz do carro", "aplicativo do carro", "conectividade bluetooth",
    "wifi no carro", "carregador sem fio", "piloto automatico adaptativo",
    "camera de estacionamento", "assistente de faixa", "sistema de som",
    "quantidade de saidas usb", "climatizacao do banco", "teto solar disponivel",
    "qual o nome do carro", "historico da marca", "onde e fabricado",
    "data de lancamento do modelo", "proxima geracao do modelo",
    "recall do veiculo", "reclamacoes no procon", "nota do euro ncap",
    "nota de seguranca", "opiniao sobre o design", "cor da lataria mais vendida",
    "melhor configuracao para familia", "vale a pena comprar",
]

PREFIXOS_RUIDO = [
    "", "", "", "qual e o ", "qual a ", "qual o ", "informe a ", "informe o ",
    "gostaria de saber a ", "gostaria de saber o ", "me diga a ", "me diga o ",
    "quero saber sobre ", "buscar ", "pesquisar ", "valor de ", "dado de ",
]

TIPOS_LETRA = ["original", "lower", "upper", "title"]


def remover_acentos(texto: str) -> str:
    forma_normalizada = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in forma_normalizada if not unicodedata.combining(c))


def aplicar_typo(texto: str, rng: random.Random) -> str:
    if len(texto) < 4:
        return texto
    posicao = rng.randrange(1, len(texto) - 1)
    tipo = rng.choice(["trocar", "remover", "duplicar", "vizinho"])
    lista = list(texto)
    if tipo == "trocar":
        lista[posicao] = rng.choice("aeiourstlnm")
    elif tipo == "remover":
        del lista[posicao]
    elif tipo == "duplicar":
        lista.insert(posicao, lista[posicao])
    elif tipo == "vizinho":
        lista[posicao], lista[posicao - 1] = lista[posicao - 1], lista[posicao]
    return "".join(lista)


def gerar_variacoes_atributo(sementes, unidades, rng: random.Random, quantidade: int):
    variacoes = set()
    tentativas = 0
    while len(variacoes) < quantidade and tentativas < quantidade * 20:
        tentativas += 1
        base = rng.choice(sementes)
        prefixo = rng.choice(PREFIXOS_RUIDO)
        texto = f"{prefixo}{base}"

        if unidades and rng.random() < 0.35:
            texto = f"{texto} {rng.choice(unidades)}"

        if rng.random() < 0.5:
            texto = remover_acentos(texto)

        estilo = rng.choice(TIPOS_LETRA)
        if estilo == "lower":
            texto = texto.lower()
        elif estilo == "upper":
            texto = texto.upper()
        elif estilo == "title":
            texto = texto.title()

        if rng.random() < 0.15:
            texto = aplicar_typo(texto, rng)

        texto = texto.strip()
        if texto:
            variacoes.add(texto)

    return list(variacoes)[:quantidade]


def gerar_variacoes_fora_catalogo(rng: random.Random, quantidade: int):
    return gerar_variacoes_atributo(FORA_DO_CATALOGO_SEMENTES, [], rng, quantidade)


def montar_dataset_limpo(seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = random.Random(seed)
    linhas = []
    for atributo, config in CATALOGO.items():
        quantidade = rng.randint(MIN_VARIACOES, MAX_VARIACOES)
        variacoes = gerar_variacoes_atributo(config["sementes"], config["unidades"], rng, quantidade)
        for texto in variacoes:
            linhas.append({"texto": texto, "atributo": atributo})

    quantidade_fora = rng.randint(180, 260)
    for texto in gerar_variacoes_fora_catalogo(rng, quantidade_fora):
        linhas.append({"texto": texto, "atributo": "fora_do_catalogo"})

    df = pd.DataFrame(linhas)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def injetar_sujeira(df: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = random.Random(seed + 1)
    df = df.copy()

    n_nulos = max(1, int(len(df) * 0.015))
    indices_nulos = rng.sample(range(len(df)), n_nulos)
    for i in indices_nulos:
        df.loc[i, "texto"] = None if rng.random() < 0.6 else "   "

    n_duplicatas = max(1, int(len(df) * 0.03))
    duplicatas = df.dropna(subset=["texto"]).sample(n=n_duplicatas, random_state=seed, replace=True)
    df = pd.concat([df, duplicatas], ignore_index=True)

    n_inconsistentes = max(1, int(len(df) * 0.01))
    candidatos = df.dropna(subset=["texto"]).sample(n=n_inconsistentes, random_state=seed + 2).index
    atributos_possiveis = list(CATALOGO.keys())
    for i in candidatos:
        atual = df.loc[i, "atributo"]
        outros = [a for a in atributos_possiveis if a != atual]
        df.loc[i, "atributo"] = rng.choice(outros)

    textos_outliers = [
        "asjdklaskjdlaksjdlk", "??????????", "12345678900", "!!!???...",
        "a" * 40, "texto completamente aleatorio sem relacao nenhuma com carros nem nada",
        "", "sdlkfjsldkfjslkdfj carro motor xpto 999",
    ]
    linhas_outliers = pd.DataFrame({
        "texto": textos_outliers,
        "atributo": ["fora_do_catalogo"] * len(textos_outliers),
    })
    df = pd.concat([df, linhas_outliers], ignore_index=True)

    df = df.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)
    return df


def gerar_e_salvar(caminho_saida: str = "dados/dataset_bruto.csv", seed: int = RANDOM_STATE) -> pd.DataFrame:
    df_limpo = montar_dataset_limpo(seed)
    df_bruto = injetar_sujeira(df_limpo, seed)
    Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
    df_bruto.to_csv(caminho_saida, index=False)
    return df_bruto


if __name__ == "__main__":
    dataset = gerar_e_salvar()
    print(f"Dataset bruto salvo com {len(dataset)} linhas.")
    print(dataset["atributo"].value_counts())
