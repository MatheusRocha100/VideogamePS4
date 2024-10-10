
# Importando as bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_extras.metric_cards import style_metric_cards
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
from scipy import stats
from fpdf import FPDF
import base64



# Configurando a paleta de cores
custom_palette = ['#025963', '#0ACE7F', '#67D7A9', '#A2D2FF', '#A2D2FF']
sns.set_palette(custom_palette)

# Configurando o site
st.title('Análise de Vendas de Videogames')

# Carregando a base de dados
Base_Dados = pd.read_csv('PS4_GamesSales.csv', encoding='latin-1')

Base_Dados = Base_Dados.dropna(subset=['Year'])

Base_Dados['Year'] = Base_Dados['Year'].astype(int)

# Removendo os anos de 2019 e 2020
Base_Dados = Base_Dados[~Base_Dados['Year'].isin([2019, 2020])]

# Filtro de anos na barra lateral
st.sidebar.title("🔎Filtros")
start_year, end_year = st.sidebar.slider(
    "Selecione o intervalo de anos",
    min_value=int(Base_Dados['Year'].min()), 
    max_value=int(Base_Dados['Year'].max()), 
    value=(int(Base_Dados['Year'].min()), int(Base_Dados['Year'].max())),
    step=1
)



traducao_generos = {
    'Action': 'Ação',
    'Adventure': 'Aventura',
    'Role-Playing': 'RPG',
    'Simulation': 'Simulação',
    'Strategy': 'Estratégia',
    'Sports': 'Esportes',
    'Racing': 'Corrida',
    'Shooter': 'Tiro',
    'Fighting': 'Luta',
    'Puzzle': 'Quebra-cabeça',
    'Platform': 'Plataforma',
    'Music': 'Música',
    'Misc': 'Diversos',
    'Action-Adventure': 'Ação-Aventura',
    'MMO': 'MMO',
    'Party': 'Festa',
    'Visual Novel': 'Visual Novel'
    }


# Traduzindo os gêneros na Base de Dados
Base_Dados['Genre Traduzido'] = Base_Dados['Genre'].map(traducao_generos)

# Filtro de gêneros de videogames
selected_genres = st.sidebar.multiselect("Selecione os Gêneros", options=Base_Dados['Genre Traduzido'].unique())

# Filtro de videogames
selected_games = st.sidebar.multiselect("Selecione os Videogames", options=Base_Dados['Game'].unique())

# Lista de continentes
continents = ['North America', 'Europe', 'Japan', 'Rest of World']

# Dicionário de continentes traduzidos
continent_translation = {
    'North America': 'América do Norte',
    'Europe': 'Europa',
    'Japan': 'Japão',
    'Rest of World': 'Resto do Mundo'
}

# Traduzindo os continentes
translated_continents = list(continent_translation.values())

# Filtro de continentes
selected_translated_continent = st.sidebar.selectbox("Selecione a Região", options=translated_continents)

# Mapeando o nome selecionado de volta para o nome original em inglês
selected_continent = [key for key, value in continent_translation.items() if value == selected_translated_continent][0]

# Filtrando os dados com base nos anos
filtered_data = Base_Dados[(Base_Dados['Year'] >= start_year) & (Base_Dados['Year'] <= end_year)]

# Aplicando o filtro de gêneros somente se algum gênero for selecionado
if selected_genres:
    filtered_data = filtered_data[filtered_data['Genre Traduzido'].isin(selected_genres)]

# Aplicando o filtro de videogames somente se algum jogo for selecionado
if selected_games:
    filtered_data = filtered_data[filtered_data['Game'].isin(selected_games)]

# Aplicando o filtro de continente
filtered_data_continent = filtered_data[['Year', selected_continent]].copy()
filtered_data_continent = filtered_data_continent.rename(columns={selected_continent: 'Sales'})

# Criando as abas
tab1, tab2, tab3, tab4, tab5= st.tabs(["Análises", "Base de Dados - EDA", "Dashboard", "Pipeline de Dados", "Contato"])


with tab1:
    
    # Encontrando o jogo mais vendido dentro dos filtros aplicados
    if not filtered_data.empty:
        game_most_sold = filtered_data.loc[filtered_data['Global'].idxmax()]
    else:
        game_most_sold = None

    # Calculando o continente com mais vendas
    if not filtered_data.empty:
        sales_by_continent = {
            'América do Norte': filtered_data['North America'].sum(),
            'Europa': filtered_data['Europe'].sum(),
            'Japão': filtered_data['Japan'].sum(),
            'Resto do Mundo': filtered_data['Rest of World'].sum()
        }
        top_continent = max(sales_by_continent, key=sales_by_continent.get)
        top_sales = sales_by_continent[top_continent]
    else:
        top_continent = "N/A"
        top_sales = 0

    # Exibindo os cartões lado a lado
    col1, col2 = st.columns(2)

    with col1:
        if game_most_sold is not None:
            st.metric(label="Jogo Mais Vendido", value=game_most_sold['Game'],delta=f"{round(game_most_sold['Global'], 2)} mi unidades")
        else:
            st.metric(label="Jogo Mais Vendido", value="N/A", delta="0 mi unidades")

    with col2:
        st.metric(label="Região com Mais Vendas", value=top_continent, delta=f"{round(top_sales, 2)} mi unidades")

    # Aplicando estilo ao cartão de métrica
    style_metric_cards(
        background_color="#FFFFFF",  
        border_radius_px=10,  
        border_color="#0ACE7F"
    )


    def salvar_grafico(figura, nome_arquivo):
        caminho_arquivo = f"{nome_arquivo}.png"
        figura.savefig(caminho_arquivo, format='png')
        return caminho_arquivo
    
    # Gráfico de Barras
    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))  # Ajuste do tamanho do gráfico
    sns.barplot(data=filtered_data, x='Year', y='Global', ci=None, color='#0ACE7F', estimator=sum, ax=ax_bar, width=0.75)
    ax_bar.set_title('Vendas Globais em (mi) Unidades', loc='left', fontsize=18)
    ax_bar.set_ylabel('')  # Removendo o título do eixo y
    ax_bar.set_xlabel('')  # Removendo o título do eixo x
    st.pyplot(fig_bar)

    salvar_grafico(fig_bar, 'grafico_barras_globais')

    st.write("\n" * 10)

    # Gráfico de Boxplot
    fig_box, ax_box = plt.subplots(figsize=(10, 5))  # Ajuste do tamanho do gráfico
    sns.boxplot(data=filtered_data, x='Year', y='Global', ax=ax_box, fliersize=5, flierprops=dict(markerfacecolor='#67D7A9', marker='o'))  # Ajustando a cor das bolinhas
    ax_box.set_title('Distribuição de Vendas Globais (mi)', loc='left', fontsize=18)
    ax_box.set_xlabel('')  # Removendo o título do eixo x
    ax_box.set_ylabel('')  # Removendo o título do eixo y
    st.pyplot(fig_box)

    salvar_grafico(fig_box, 'grafico_boxplot')


    st.write("\n" * 10)

    # Gráfico de Barras Empilhadas
    Analise = filtered_data.groupby('Year').sum().reset_index()
    America = Analise['North America'] / Analise['Global'] * 100
    Europa = Analise['Europe'] / Analise['Global'] * 100
    Japao = Analise['Japan'] / Analise['Global'] * 100
    Mundo = Analise['Rest of World'] / Analise['Global'] * 100

    plt.figure(figsize=(10, 5))
    Largura_Barra = 0.85
    Rotulos = Analise['Year']
    Grupos = list(range(len(Rotulos)))

    plt.title('Distribuição de Vendas Globais (mi) por Região', loc='left', fontsize=18)
    plt.bar(Grupos, America, width=Largura_Barra, color=custom_palette[0], edgecolor='white')
    plt.bar(Grupos, Europa, bottom=America, width=Largura_Barra, color=custom_palette[1], edgecolor='white')
    plt.bar(Grupos, Japao, bottom=America + Europa, width=Largura_Barra, color=custom_palette[2], edgecolor='white')
    plt.bar(Grupos, Mundo, bottom=America + Europa + Japao, width=Largura_Barra, color=custom_palette[3], edgecolor='white')

    plt.xticks(Grupos, Rotulos)
    plt.xlabel('Ano')
    plt.ylabel('')
    plt.legend(['América N', 'Europa', 'Japão', 'Mundo'], loc='upper left', bbox_to_anchor=(0.15, -0.1), ncol=4)
    st.pyplot(plt)

    plt.savefig('grafico_barras_empilhadas.png') 
    salvar_grafico(plt.gcf(), 'grafico_barras_empilhadas')


    col1, col2 = st.columns([3, 1]) 

    # Filtrando os dados com base nos anos e continente
    filtered_data_regression = Base_Dados[(Base_Dados['Year'] >= start_year) & (Base_Dados['Year'] <= end_year)]
    filtered_data_continent_regression = filtered_data_regression[['Year', selected_continent]].copy()
    filtered_data_continent_regression = filtered_data_continent_regression.rename(columns={selected_continent: 'Sales'})

    # Removendo outliers usando o Z-score para as vendas
    z_scores = np.abs(stats.zscore(filtered_data_continent_regression['Sales']))
    filtered_data_continent_regression = filtered_data_continent_regression[z_scores < 3]  # Considerando Z-score acima de 3 como outlier

    # Separando os dados de treino e teste
    X = filtered_data_continent_regression[['Year']]  # Variável independente (Ano)
    y = filtered_data_continent_regression['Sales']  # Variável dependente (Vendas do continente selecionado)

    # Dividindo os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Criando o modelo de regressão linear
    regressor = LinearRegression()

    # Treinando o modelo
    regressor.fit(X_train, y_train)

    # Fazendo previsões no conjunto de teste
    y_pred = regressor.predict(X_test)

    # Prevendo para os próximos dois anos
    future_years = np.array([[end_year + 1], [end_year + 2]])  # Próximos dois anos
    future_predictions = regressor.predict(future_years)

    # Gráfico de dispersão com a linha de regressão
    fig_regression, ax_regression = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=X_test['Year'], y=y_test, ax=ax_regression, color='#0ACE7F', edgecolor='black', linewidth=1, label='Unidades')  # Adiciona o contorno preto
    sns.lineplot(x=X_test['Year'], y=y_pred, ax=ax_regression, color='#025963', label='Linha de Regressão')

    # Adicionando os anos futuros e as previsões ao gráfico
    sns.scatterplot(x=future_years.flatten(), y=future_predictions, ax=ax_regression, color='#FF5733',edgecolor='black', label='Previsão Futura')  # Pontos para previsão futura
    sns.lineplot(x=np.concatenate([X_test['Year'].values, future_years.flatten()]), 
                y=np.concatenate([y_pred, future_predictions]), 
                ax=ax_regression, color='#025963', linestyle='--', label='Projeção')

    # Título do Gráfico
    ax_regression.set_title(f"Regressão Linear das Vendas - {selected_translated_continent} por Ano", loc='left', fontsize=18)

    # Removendo os títulos dos eixos
    ax_regression.set_xlabel('Ano')
    ax_regression.set_ylabel('')


    # Exibindo as previsões futuras
    for i, year in enumerate(future_years.flatten()):
        ax_regression.text(year, future_predictions[i] + 0.02,  # Ajuste aqui, aumentando o valor da coordenada y
                        f"{round(future_predictions[i], 2)}", 
                        color="#FF5733", fontsize=12, ha='center')


    st.pyplot(fig_regression)

    salvar_grafico(fig_regression, 'grafico_regressao_linear')


    # Mapa de Calor
    fig_heat, ax_heat = plt.subplots(figsize=(10, 5))  # Ajuste do tamanho do gráfico
    custom_cmap = sns.color_palette(['#67D7A9', '#0ACE7F','#1E8568', '#025963'], as_cmap=True)
    heatmap_data = pd.pivot_table(filtered_data, values='Global', index='Genre', columns='Year', aggfunc=np.sum, fill_value=0)
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap=custom_cmap, linewidths=.5, ax=ax_heat)
    ax_heat.set_xlabel('')
    ax_heat.set_ylabel('')
    ax_heat.set_title('Vendas Globais (mi) por Gênero e Ano', loc='left', fontsize=18)
    st.pyplot(fig_heat)

    salvar_grafico(fig_heat, 'grafico_mapa_calor')

    def gerar_pdf(imagens, nome_arquivo_pdf, start_year, end_year, selected_genres, selected_games, selected_translated_continent):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
    
        # Adiciona uma página
        pdf.add_page()
    
        # Adiciona título ao PDF
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, txt="Relatório de Vendas de Videogames", ln=True, align="C")
    
        # Adiciona uma seção para os filtros
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, txt="Filtros Aplicados:", ln=True, align="L")
    
        pdf.set_font("Arial", "", 12)
        pdf.cell(200, 10, txt=f"Intervalo de Anos: {start_year} - {end_year}", ln=True, align="L")
    
        # Mostra os gêneros selecionados
        if selected_genres:
            pdf.cell(200, 10, txt=f"Gêneros Selecionados: {', '.join(selected_genres)}", ln=True, align="L")
        else:
            pdf.cell(200, 10, txt="Gêneros Selecionados: Todos", ln=True, align="L")
    
        # Mostra os videogames selecionados
        if selected_games:
            pdf.cell(200, 10, txt=f"Videogames Selecionados: {', '.join(selected_games)}", ln=True, align="L")
        else:
            pdf.cell(200, 10, txt="Videogames Selecionados: Todos", ln=True, align="L")
    
        # Mostra o continente selecionado
        pdf.cell(200, 10, txt=f"Região Selecionada: {selected_translated_continent}", ln=True, align="L")
    
        # Adiciona as imagens dos gráficos no PDF
        for imagem in imagens:
            pdf.image(imagem, x=10, y=None, w=190)  # Ajusta as coordenadas x, y e a largura (w) da imagem
    
        # Salva o PDF
        pdf.output(nome_arquivo_pdf)


    # Lista com os gráficos salvos
    imagens_para_pdf = ['grafico_barras_globais.png', 'grafico_boxplot.png', 'grafico_barras_empilhadas.png', 'grafico_regressao_linear.png', 'grafico_mapa_calor.png']

    # Gerar o PDF
    nome_pdf = "relatorio_vendas_videogames.pdf"

    # Chamada da função gerar_pdf com os filtros
    gerar_pdf(imagens_para_pdf, nome_pdf, start_year, end_year, selected_genres, selected_games, selected_translated_continent)

    # Exibir link para download na barra lateral
    with st.sidebar:
        st.write("")
        with open(nome_pdf, "rb") as pdf_file:
            PDFbytes = pdf_file.read()
            b64 = base64.b64encode(PDFbytes).decode('utf-8')
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="{nome_pdf}">Baixar relatório em PDF</a>'
            st.markdown(href, unsafe_allow_html=True)
 

    # Cópia da Base de Dados sem valores ausentes na coluna 'Year'
    Base_Dados_sem_tratativa = pd.read_csv('PS4_GamesSales.csv', encoding='latin-1')



with tab2:

    # Exibir a Base de Dados
    st.subheader("🎲Base de Dados:")
    st.dataframe(Base_Dados_sem_tratativa[['Year', 'Game', 'Genre', 'Global', 'North America', 'Europe', 'Japan', 'Rest of World']])

    # Informações sobre a Base de Dados
    st.write("Dimensões do dataset:", Base_Dados_sem_tratativa.shape)

    def df_info(df):
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Non-Null Count": df.notnull().sum(),
            "Dtype": df.dtypes
        }).reset_index(drop=True)
        return info_df

    # Tabela de informações
    info_df = df_info(Base_Dados_sem_tratativa)

    # Exibindo informações no Streamlit
    st.subheader("📊 Informações da Base:")
    st.dataframe(info_df)

    # Estatísticas descritivas
    st.subheader("📈Estatísticas da Base:")
    st.write(Base_Dados_sem_tratativa.describe())

    # Valores ausentes por coluna
    valores_ausentes = Base_Dados_sem_tratativa.isna().sum()

    # Exibe os valores ausentes sem cabeçalho
    st.subheader("Valores ausentes por coluna:")
    st.table(pd.DataFrame(valores_ausentes.values, index=valores_ausentes.index, columns=[""]).T)

    # Dicionário de tradução para gêneros
    traducao_generos = {
        'Action': 'Ação',
        'Adventure': 'Aventura',
        'Role-Playing': 'RPG',
        'Simulation': 'Simulação',
        'Strategy': 'Estratégia',
        'Sports': 'Esportes',
        'Racing': 'Corrida',
        'Shooter': 'Tiro',
        'Fighting': 'Luta',
        'Puzzle': 'Quebra-cabeça',
        'Platform': 'Plataforma',
        'Music': 'Música',
        'Misc': 'Diversos',
        'Action-Adventure': 'Ação-Aventura',
        'MMO': 'MMO',
        'Party': 'Festa',
        'Visual Novel': 'Visual Novel'
    }

    # Dicionário de cores para cada gênero traduzido
    cores_generos = {
        'Ação': '#025963',
        'Aventura': '#025963',
        'RPG': '#025963',
        'Simulação': '#67D7A9',
        'Estratégia': '#67D7A9',
        'Esportes': '#0ACE7F',
        'Corrida': '#0ACE7F',
        'Tiro': '#0ACE7F',
        'Luta': '#67D7A9',
        'Quebra-cabeça': '#A2D2FF',
        'Plataforma': '#0ACE7F',
        'Música': '#A2D2FF',
        'Diversos': '#025963',
        'Ação-Aventura': '#67D7A9',
        'MMO': '#A2D2FF',
        'Festa': '#A2D2FF',
        'Visual Novel': '#A2D2FF'
        }

    # Traduzir os gêneros na Base de Dados
    Base_Dados_sem_tratativa['Genre Traduzido'] = Base_Dados_sem_tratativa['Genre'].map(traducao_generos)

    # Contagem de jogos por gênero, ordenando do maior para o menor
    genero_ordenado = Base_Dados_sem_tratativa['Genre Traduzido'].value_counts()

    # Criar a lista de cores para o gráfico com base no dicionário de cores
    cores_ordenadas = [cores_generos[genero] for genero in genero_ordenado.index]

    # Criar o gráfico de barras com as cores aplicadas a partir do dicionário
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=genero_ordenado.index, y=genero_ordenado.values, palette=cores_ordenadas)

    # Configurações do gráfico
    ax.set_title('Distribuição de Videogames por Gênero', loc='left', fontsize=18)
    ax.set_xticklabels(genero_ordenado.index, rotation=45, ha='right')
    ax.set_xlabel('')  # Remover título do eixo x
    ax.set_ylabel('')  # Remover título do eixo y

    # Exibir o gráfico
    st.pyplot(fig)

    # Gráfico de linha com continentes e paleta de cores definida sem áreas sombreadas
    fig, ax = plt.subplots(figsize=(10, 5))

    # Adicionar linhas para cada continente com cores personalizadas, diferentes marcadores e sem áreas sombreadas
    sns.lineplot(data=Base_Dados, x='Year', y='North America', color='#0ACE7F', estimator=sum, marker='o', label='América do Norte', ci=None)  # Cor: #025963, marcador: o
    sns.lineplot(data=Base_Dados, x='Year', y='Europe', color='#025963', estimator=sum, marker='s', label='Europa', ci=None)  # Cor: #67D7A9, marcador: quadrado
    sns.lineplot(data=Base_Dados, x='Year', y='Japan', color='#A2D2FF', estimator=sum, marker='^', label='Japão', ci=None)  # Cor: #0ACE7F, marcador: triângulo
    sns.lineplot(data=Base_Dados, x='Year', y='Rest of World', color='#67D7A9', estimator=sum, marker='D', label='Resto do Mundo', ci=None)  # Cor: #A2D2FF, marcador: diamante

    # Ajustar títulos e rótulos
    ax.set_title('Vendas por Região ao Longo dos Anos', loc='left', fontsize=18)
    ax.set_xlabel('')  # Remover título do eixo x
    ax.set_ylabel('')  # Remover título do eixo y

    # Adicionar legenda para os continentes
    ax.legend(title='Regiões')

    # Exibir o gráfico no Streamlit
    st.pyplot(fig)



with tab3:

    # Organizando os Gráficos em Colunas
    col1, col2 = st.columns(2)

    with col1:
        # Gráfico de Barras
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        sns.barplot(data=filtered_data, x='Year', y='Global', ci=None, color='#0ACE7F', estimator=sum, ax=ax_bar)
        ax_bar.set_title('Vendas Globais em (mi) Unidades', loc='left', fontsize=18)
        ax_bar.set_ylabel('')
        ax_bar.set_xlabel('')
        st.pyplot(fig_bar)
    
    with col2:
        # Gráfico de Boxplot
        fig_box, ax_box = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=filtered_data, x='Year', y='Global', ax=ax_box, fliersize=5, flierprops=dict(markerfacecolor='#67D7A9', marker='o'))
        ax_box.set_title('Distribuição de Vendas Globais (mi)', loc='left', fontsize=18)
        ax_box.set_xlabel('')
        ax_box.set_ylabel('')
        st.pyplot(fig_box)

    col3, col4 = st.columns(2)

    with col3:
        # Gráfico de Colunas Clusterizadas
        Analise = filtered_data.groupby('Year').sum().reset_index()
        America = Analise['North America'] / Analise['Global'] * 100
        Europa = Analise['Europe'] / Analise['Global'] * 100
        Japao = Analise['Japan'] / Analise['Global'] * 100
        Mundo = Analise['Rest of World'] / Analise['Global'] * 100

        plt.figure(figsize=(10, 6))
        Largura_Barra = 0.85
        Rotulos = Analise['Year']
        Grupos = list(range(len(Rotulos)))

        plt.title('Distribuição de Vendas Globais (mi) por Região', loc='left', fontsize=18)
        plt.bar(Grupos, America, width=Largura_Barra, color=custom_palette[0], edgecolor='white')
        plt.bar(Grupos, Europa, bottom=America, width=Largura_Barra, color=custom_palette[1], edgecolor='white')
        plt.bar(Grupos, Japao, bottom=America + Europa, width=Largura_Barra, color=custom_palette[2], edgecolor='white')
        plt.bar(Grupos, Mundo, bottom=America + Europa + Japao, width=Largura_Barra, color=custom_palette[3], edgecolor='white')

        plt.xticks(Grupos, Rotulos)
        plt.xlabel('Ano')
        plt.ylabel('')
        plt.legend(['América N', 'Europa', 'Japão', 'Mundo'], loc='upper left', bbox_to_anchor=(0.15, -0.1), ncol=4)
        st.pyplot(plt)

    with col4:
        # Gráfico de Regressão Linear
        filtered_data_regression = Base_Dados[(Base_Dados['Year'] >= start_year) & (Base_Dados['Year'] <= end_year)]
        filtered_data_continent_regression = filtered_data_regression[['Year', selected_continent]].copy()
        filtered_data_continent_regression = filtered_data_continent_regression.rename(columns={selected_continent: 'Sales'})

        # Removendo outliers usando o Z-score para as vendas
        z_scores = np.abs(stats.zscore(filtered_data_continent_regression['Sales']))
        filtered_data_continent_regression = filtered_data_continent_regression[z_scores < 3]  # Considerando Z-score acima de 3 como outlier

        # Separando os dados de treino e teste
        X = filtered_data_continent_regression[['Year']]  # Variável independente (Ano)
        y = filtered_data_continent_regression['Sales']  # Variável dependente (Vendas do continente selecionado)

        # Dividindo os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Criando o modelo de regressão linear
        regressor = LinearRegression()

        # Treinando o modelo
        regressor.fit(X_train, y_train)

        # Fazendo previsões no conjunto de teste
        y_pred = regressor.predict(X_test)

        # Prevendo para os próximos dois anos
        future_years = np.array([[end_year + 1], [end_year + 2]])  # Próximos dois anos
        future_predictions = regressor.predict(future_years)

        # Gráfico de dispersão com a linha de regressão
        fig_regression, ax_regression = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=X_test['Year'], y=y_test, ax=ax_regression, color='#0ACE7F', edgecolor='black', linewidth=1, label='Unidades')  # Adiciona o contorno preto
        sns.lineplot(x=X_test['Year'], y=y_pred, ax=ax_regression, color='#025963', label='Linha de Regressão')

        # Adicionando os anos futuros e as previsões ao gráfico
        sns.scatterplot(x=future_years.flatten(), y=future_predictions, ax=ax_regression, color='#FF5733',edgecolor='black', label='Previsão Futura')  # Pontos para previsão futura
        sns.lineplot(x=np.concatenate([X_test['Year'].values, future_years.flatten()]), 
                    y=np.concatenate([y_pred, future_predictions]), 
                    ax=ax_regression, color='#025963', linestyle='--', label='Projeção')

        # Definir o título com o mesmo tamanho de fonte
        ax_regression.set_title(f"Regressão Linear das Vendas - {selected_translated_continent} por Ano", loc='left', fontsize=18)

        # Remover o título dos eixos
        ax_regression.set_xlabel('Ano')
        ax_regression.set_ylabel('')


        # Exibindo as previsões futuras
        for i, year in enumerate(future_years.flatten()):
            ax_regression.text(year, future_predictions[i] + 0.02,  # Ajuste aqui, aumentando o valor da coordenada y
                            f"{round(future_predictions[i], 2)}", 
                            color="#FF5733", fontsize=12, ha='center')


        st.pyplot(fig_regression)



   # Lista completa de códigos ISO-3 de países
    all_countries = [
        'AFG', 'ALB', 'DZA', 'ASM', 'AND', 'AGO', 'AIA', 'ATA', 'ATG', 'ARG', 'ARM', 'ABW', 'AUS', 'AUT', 'AZE', 
        'BHS', 'BHR', 'BGD', 'BRB', 'BLR', 'BEL', 'BLZ', 'BEN', 'BMU', 'BTN', 'BOL', 'BES', 'BIH', 'BWA', 'BRA', 
        'VGB', 'BRN', 'BGR', 'BFA', 'BDI', 'KHM', 'CMR', 'CAN', 'CPV', 'CYM', 'CAF', 'TCD', 'CHL', 'CHN', 'COL', 
        'COM', 'COG', 'COD', 'COK', 'CRI', 'CIV', 'HRV', 'CUB', 'CUW', 'CYP', 'CZE', 'DNK', 'DJI', 'DMA', 'DOM', 
        'ECU', 'EGY', 'SLV', 'GNQ', 'ERI', 'EST', 'SWZ', 'ETH', 'FJI', 'FIN', 'FRA', 'PYF', 'GAB', 'GMB', 'GEO', 
        'DEU', 'GHA', 'GIB', 'GRC', 'GRL', 'GRD', 'GUM', 'GTM', 'GGY', 'GIN', 'GNB', 'GUY', 'HTI', 'HND', 'HKG', 
        'HUN', 'ISL', 'IND', 'IDN', 'IRN', 'IRQ', 'IRL', 'IMN', 'ISR', 'ITA', 'JAM', 'JPN', 'JEY', 'JOR', 'KAZ', 
        'KEN', 'KIR', 'KWT', 'KGZ', 'LAO', 'LVA', 'LBN', 'LSO', 'LBR', 'LBY', 'LIE', 'LTU', 'LUX', 'MAC', 'MDG', 
        'MWI', 'MYS', 'MDV', 'MLI', 'MLT', 'MHL', 'MRT', 'MUS', 'MEX', 'FSM', 'MDA', 'MCO', 'MNG', 'MNE', 'MSR', 
        'MAR', 'MOZ', 'MMR', 'NAM', 'NRU', 'NPL', 'NLD', 'NCL', 'NZL', 'NIC', 'NER', 'NGA', 'NIU', 'PRK', 'MKD', 
        'NOR', 'OMN', 'PAK', 'PLW', 'PSE', 'PAN', 'PNG', 'PRY', 'PER', 'PHL', 'POL', 'PRT', 'PRI', 'QAT', 'ROU', 
        'RUS', 'RWA', 'KNA', 'LCA', 'VCT', 'WSM', 'SMR', 'STP', 'SAU', 'SEN', 'SRB', 'SYC', 'SLE', 'SGP', 'SXM', 
        'SVK', 'SVN', 'SLB', 'SOM', 'ZAF', 'KOR', 'SSD', 'ESP', 'LKA', 'SDN', 'SUR', 'SWE', 'CHE', 'SYR', 'TWN', 
        'TJK', 'TZA', 'THA', 'TLS', 'TGO', 'TON', 'TTO', 'TUN', 'TUR', 'TKM', 'TUV', 'UGA', 'UKR', 'ARE', 'GBR', 
        'USA', 'URY', 'UZB', 'VUT', 'VEN', 'VNM', 'YEM', 'ZMB', 'ZWE'
    ]

    # Países já usados nos continentes "North America", "Europe" e "Asia"
    included_countries = set(['USA', 'CAN', 'MEX', 'DEU', 'FRA', 'GBR', 'ITA', 'ESP', 'JPN','NOR','PRT','IRL','CHER','BEL','NLD',
                              'BLR','FIN','UKR','CZE','HRV','SRB','MKD','GRC','BGR','ROU','LTU','POL','HUN','SWE','EST','LVA',
                              'AUT','CHE','NLD','DNK','SVK','ALB','BIH','MDA','SVN','MNE','LUX','CYP'])

    # Países que fazem parte do "Rest of World"
    remaining_countries = set(all_countries) - included_countries

    # Dados de vendas por continente (agregados)
    continent_sales = {
        'North America': filtered_data['North America'].sum(),
        'Europe': filtered_data['Europe'].sum(),
        'Japan': filtered_data['Japan'].sum(),
        'Rest of World': filtered_data['Rest of World'].sum()
    }

    # Mapeamento de países por continente
    country_mapping = {
        'North America': ['USA', 'CAN', 'MEX'],  # Países da América do Norte
        'Europe': ['DEU', 'FRA', 'GBR', 'ITA', 'ESP','NOR','PRT','IRL','CHER','BEL','NLD'
        'BLR','FIN','UKR','CZE','HRV','SRB','MKD','GRC','BGR','ROU','LTU','POL','HUN','SWE','EST','LVA',
        'AUT','CHE','NLD','DNK','SVK','ALB','BIH','MDA','SVN','MNE','LUX','CYP'],  # Países da Europa
        'Japan': ['JPN'],  # Países da Ásia
        'Rest of World': list(remaining_countries)  # Países do resto do mundo
    }
    
    # Preparar os dados para o mapa
    data = pd.DataFrame({
        'Country': sum(country_mapping.values(), []),
        'Continent': ['North America'] * len(country_mapping['North America']) + 
                    ['Europe'] * len(country_mapping['Europe']) + 
                    ['Japan'] * len(country_mapping['Japan']) + 
                    ['Rest of World'] * len(country_mapping['Rest of World']),
        'Sales': [continent_sales['North America']] * len(country_mapping['North America']) +
                [continent_sales['Europe']] * len(country_mapping['Europe']) +
                [continent_sales['Japan']] * len(country_mapping['Japan']) +
                [continent_sales['Rest of World']] * len(country_mapping['Rest of World'])
    })

    # Paleta de cores personalizada
    custom_palette = {
        'North America': '#0ACE7F',
        'Europe': '#025963',
        'Japan': '#67D7A9',
        'Rest of World': '#A2D2FF' 
    }

    # Mapeamento de nomes dos continentes em português
    continent_translation = {
        'North America': 'América do Norte',
        'Europe': 'Europa',
        'Japan': 'Japão',
        'Rest of World': 'Resto do Mundo'
    }

    # Adicionar uma nova coluna "Continente_PT" com os nomes traduzidos
    data['Continente_PT'] = data['Continent'].map(continent_translation)

    # Renomear a coluna 'Sales' para 'Vendas'
    data['Vendas'] = data['Sales']

    # Adicionar uma nova coluna "País"
    data['País'] = data['Country']

    # Criar o mapa coroplético usando Plotly Express
    fig = px.choropleth(data_frame=data,
                        locations='Country',  # Usa os códigos ISO-3 dos países
                        locationmode='ISO-3',  # Especifica o modo de localização como ISO-3
                        color='Continent',  # Define a cor pelos continentes
                        hover_name='Continente_PT',  # Mostra apenas o nome do continente em português ao passar o mouse
                        hover_data={'Vendas': True, 'País': True, 'Country': False, 'Continent': False},  # Exibe "Vendas" e "País" e oculta o nome do país original (Country)
                        color_discrete_map=custom_palette  # Define a paleta de cores personalizada
                        )


    # Ajustando o layout do mapa
    fig.update_layout(
        title="",  
        title_font=dict(size=18),  
        title_x=0,  
        title_y=0.95,  
        width=2000,  
        height=400,  
        margin={"r": 0, "t": 15, "l": 0, "b": 0},  
        coloraxis_showscale=False,  # Removes the color scale bar
        showlegend=False,  # Removes the color legend
        geo=dict(
            lataxis_range=[-58, 90]  
    )
    )

    st.plotly_chart(fig)


with tab4:

    # Gif do Pipeline de Dados
    gif_url = "https://raw.githubusercontent.com/MatheusRocha100/VideogamePS4/main/Pipeline-de-Dados-7-_online-video-cutter.com_-_1_.gif"
    gif_html = f'<img src="{gif_url}" alt="GIF" width="400%">'
    st.markdown(gif_html, unsafe_allow_html=True)




with tab5:

    # Informações de Contato
    st.header("Dados para contato:")
    st.write("💼 [Portfólio: RBI-Dashboards](https://sites.google.com/view/matheus-rocha-de-sousa-lima?usp=sharing)")
    st.write("🔗 [LinkedIn: Matheus Rocha](https://www.linkedin.com/in/matheus-rocha-de-sousa-lima-92174a231/)")
    st.write("🐙 [GitHub: MatheusRocha100](https://github.com/MatheusRocha100)")
    
