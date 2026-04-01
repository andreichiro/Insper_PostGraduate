{% docs raw_dim_teachers %}
Tabela raw de cadastro de professores do AprendiZAP. A granularidade é um professor por linha. Ela concentra os atributos declarados no cadastro, como origem de aquisição, localização, etapa, disciplina e informações básicas sobre alunos.
{% enddocs %}

{% docs raw_fct_teachers_contents_interactions %}
Tabela raw de interações de professores com conteúdos e navegação da plataforma. A granularidade é um evento de interação por linha. Ela é a principal origem dos sinais de uso de produto, visualização, download e atividade.
{% enddocs %}

{% docs raw_fct_teachers_entries %}
Tabela raw de entradas ou sessões dos professores na plataforma. A granularidade é uma sessão por linha. Ela é a origem dos sinais de início, fim e duração de sessão.
{% enddocs %}

{% docs raw_fct_mari_ia_eventos_isso_ajudou %}
Tabela raw de feedbacks sobre a Mari IA. A granularidade é um evento relacionado à Mari por linha. Ela ajuda a reconstruir quando houve ajuda, qual foi a resposta do usuário e se a ajuda foi percebida como útil.
{% enddocs %}

{% docs raw_stg_formation %}
Tabela raw de eventos de formação. A granularidade é um registro de progresso ou conclusão de item de curso por linha. Ela é a base dos sinais de engajamento em trilhas formativas.
{% enddocs %}

{% docs raw_stg_mari_ia_conversation %}
Tabela raw de conversas com a Mari IA no formato antigo, anterior à mudança para relatórios estruturados. A granularidade é uma conversa por linha.
{% enddocs %}

{% docs raw_stg_mari_ia_reports %}
Tabela raw de relatórios estruturados das interações com a Mari IA no formato mais novo. A granularidade é um atributo reportado por linha. Ela complementa a tabela de conversas legadas.
{% enddocs %}

{% docs raw_stg_lessons %}
Catálogo raw de aulas e metadados educacionais. A granularidade é uma aula por linha. Ela permite enriquecer interações de conteúdo com nível, disciplina, unidade e indicadores pedagógicos.
{% enddocs %}

{% docs raw_hotjar_pesquisa_mobile %}
Pesquisa raw do Hotjar respondida em contexto mobile. A granularidade é uma resposta de pesquisa por linha. Ela documenta feedback de experiência, contexto do navegador e disposição do usuário em conversar com o time.
{% enddocs %}

{% docs raw_hotjar_pesquisa_desktop %}
Pesquisa raw do Hotjar respondida em contexto desktop. A granularidade é uma resposta de pesquisa por linha. Ela documenta feedback de experiência em navegação desktop.
{% enddocs %}

{% docs raw_hotjar_teste_interesse %}
Pesquisa raw do Hotjar voltada a interesse e troca entre educadores. A granularidade é uma resposta por linha. Ela registra estratégias de engajamento declaradas e interesse em discutir desafios educacionais com outros professores.
{% enddocs %}

{% docs raw_school_calendar %}
Calendário escolar raw. A granularidade é um mês, UF e rede por linha. Ele fornece dias úteis, feriados e sinalização do período escolar para enriquecer contexto temporal na base modelada.
{% enddocs %}
