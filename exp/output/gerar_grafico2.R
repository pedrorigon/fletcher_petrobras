# Carregar o pacote ggplot2
library(ggplot2)

# Obter o nome do arquivo CSV como argumento
args <- commandArgs(trailingOnly = TRUE)
nome_arquivo <- args[1]

# Ler o arquivo CSV de entrada
dados <- read.csv(nome_arquivo)

# Modificar o nome dos aplicativos mantendo apenas a parte anterior ao "."
dados$Aplicativo <- gsub("\\..*", "", dados$Aplicativo)

# Renomear as colunas conforme o arquivo CSV fornecido
colnames(dados) <- c("Aplicativo", "Tamanho", "Resultado_Medio", "Desvio_Padrao", "IC_Inferior", "IC_Superior")

# Definir a largura e altura do gráfico em polegadas
largura <- 8
altura <- 6

# Definir a densidade em dpi (alta resolução)
densidade <- 300

# Criar o gráfico com barras
grafico <- ggplot(dados, aes(x = reorder(factor(Tamanho), Tamanho), y = Resultado_Medio, fill = Aplicativo)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_errorbar(aes(ymin = Resultado_Medio - Desvio_Padrao, ymax = Resultado_Medio + Desvio_Padrao), width = 0.3, position = position_dodge(width = 0.7)) +
  labs(x = "Tamanho de Problema", y = "MSamples/s") +
  scale_y_continuous(breaks = seq(0, max(dados$Resultado_Medio), by = 100), limits = c(0, max(dados$Resultado_Medio) * 1.1)) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 10, color = "black", hjust = 0.5, vjust = 0.5),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title = element_text(size = 12, color = "black"),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    legend.position = "top",
    legend.box.spacing = unit(0.2, "cm")
  ) +
  guides(fill = guide_legend(ncol = 2))

# Salvar o gráfico em um arquivo PDF em alta definição
nome_grafico <- gsub("\\..*", "", nome_arquivo)
ggsave(paste0(nome_grafico, ".pdf"), plot = grafico, width = largura, height = altura, dpi = densidade)

