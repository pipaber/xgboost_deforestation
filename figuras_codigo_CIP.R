#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#                                                                              #
#                 Código para gráficos - estudio deforestación                 #
#                                                                              #
#             Consultora: Corina Navarrete                                     #
#            Institución: Centro Internacional de la Papa                      #
#                                                                              #
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Preparación ----
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


## Configuración de entorno
g <- gc(reset = T)
rm(list = ls())
options(scipen = 999, warn = -1)

## Paquetes necesarios necesarias
pacman::p_load(data.table, corrplot,  ggcorrplot, tidyverse, MASS, skimr,
               ggpubr, ggcorrplot, tidymodels, doParallel, bonsai, baguette,
               rules, iml, DataExplorer, plotly, ggdendro, caret, showtext,
               ggpattern, dendextend)



font_add_google("Lato", "Lt")
showtext_auto()


##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Data sets ----
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


base_pe <- fread("deforestation_dataset_2024_Peru.txt",
                 colClasses=c(UBIGEO ="character"))





## Correlación Deforestación vs Impulsores a nivel Sierra -----------------------
base_pe_sierra <- base_pe |>
  filter(Región== 'SIERRA')

CC <- as.matrix(base_pe_sierra[,c(7:33)])
rcorrDat<- Hmisc::rcorr(CC, type="spearman")
tab_cor  <- data.frame(r_value = rcorrDat$r[,1],
                       p_value = rcorrDat$P[,1],
                       n_obs = rcorrDat$n[,1]) |>
  rownames_to_column("Variables") |>
  mutate(int_sig = if_else(p_value <0.05, "sig", "no sig"))

tab_cor_sig <- tab_cor |>
  filter(int_sig == "sig")


fig_1 <- tab_cor_sig |>
  mutate(rowname = factor(Variables, levels = Variables[order(r_value)])) %>%
  ggplot(aes(x = rowname, y = r_value)) +
  geom_col_pattern(
    stat = "identity",
    pattern = 'stripe',
    fill    = '#bf8040',
    pattern_fill    = 'white',
    pattern_colour = '#bf8040',
    colour  = 'black'
  )  +
  # scale_x_discrete(labels=c('Dist. to nat. commun.', 'Dist. to rivers',
  #                           'Dist. to ANP',
  #                           'Efficiency of public spending','Dist. to mining cadastre',
  #                           'IDH','Mining','Patch density',
  #                           'Coca Crop','Temperature',
  #                           'Infrastructure','Emigrants',
  #                           'Population',
  #                           'Immigrants','Gross Domestic Product',
  #                           'Agricultural workers',
  #                           'Agricultural and livestock','River, lake or ocean',
  #                           'Yuca Root','Rainfall','Dist. to roads',
#                           'Aggregation index ','Shannon index'))+
labs(x = 'Potential predictors', y = 'Correlation with deforested area') +
  ggtitle(label = 'Association between deforestation and potential predictors', subtitle = 'Districts in Sierra | Based on uni/multi temporal data') +
  theme(plot.title = element_text(hjust = 0.5, size = 18),
        plot.subtitle = element_text(face = 'italic', hjust = 0.5, size = 16),
        axis.text.y = element_text(angle = 90, hjust = 0.5, size = 11),
        axis.text.x = element_text(size = 14, angle = 45, vjust = 1, hjust=1),
        axis.title.y = element_text(size = 16, family = "Lt"),
        axis.title.x = element_text(size = 16, family = "Lt"),
        legend.text = element_text(size = 9.5),
        plot.caption = element_text(hjust = 0, size = 8.5)) +
  theme(text = element_text(family = "Lt"))


ggsave(plot = fig_1, filename = "mapas_graficos/correlation_Sierra.png", units = 'in',
       width = 7, height = 4, dpi = 300)


## Correlación Deforestación vs Impulsores a nivel Selva-----------------------
base_pe_selva <- base_pe |>
  filter(Región== 'SELVA')

CC <- as.matrix(base_pe_selva[,c(7:33)])
rcorrDat<- Hmisc::rcorr(CC, type="spearman")
tab_cor  <- data.frame(r_value = rcorrDat$r[,1],
                       p_value = rcorrDat$P[,1],
                       n_obs = rcorrDat$n[,1]) |>
  rownames_to_column("Variables") |>
  mutate(int_sig = if_else(p_value <0.05, "sig", "no sig"))

tab_cor_sig <- tab_cor |>
  filter(int_sig == "sig")


fig_2 <- tab_cor_sig |>
  mutate(rowname = factor(Variables, levels = Variables[order(r_value)])) |>
  ggplot(aes(x = rowname, y = r_value)) +
  geom_col_pattern(
    stat = "identity",
    pattern = 'stripe',
    fill    = '#00d540',
    pattern_fill    = 'white',
    pattern_colour = '#00d540',
    colour  = 'black'
  )  +
  scale_x_discrete(labels=c('Dist. to nat. commun.', 'Patch density',
                            'Dist. to timber concessions',
                            'Dist. to ANP',
                            'Dist. to rivers','Shannon index',
                            'Coca Crop','IDH',
                            'Efficiency of public spending','Soil Moisture',
                            'Mining', 'Dist. to mining cadastre',
                            'Gross Domestic Product','Temperature',
                            'Infrastructure','Emigrants',
                            'Population',
                            'Immigrants', 'Agricultural workers',
                            'Rainfall','Distance to roads',
                            'Aggregation index ','Yuca Root',
                            'River, lake or ocean','Agricultural and livestock'))+
  labs(x = 'Potential predictors', y = 'Correlation with deforested area') +
  ggtitle(label = 'Association between deforestation and potential predictors', subtitle = 'Districts in Selva | Based on uni/multi temporal data') +
  theme(plot.title = element_text(hjust = 0.5, size = 18),
        plot.subtitle = element_text(face = 'italic', hjust = 0.5, size = 16),
        axis.text.y = element_text(angle = 90, hjust = 0.5, size = 11),
        axis.text.x = element_text(size = 14, angle = 45, vjust = 1, hjust=1),
        axis.title.y = element_text(size = 16, family = "Lt"),
        axis.title.x = element_text(size = 16, family = "Lt"),
        legend.text = element_text(size = 9.5),
        plot.caption = element_text(hjust = 0, size = 8.5)) +
  theme(text = element_text(family = "Lt"))


fig_2

ggsave(plot = fig_2, filename = "mapas_graficos/correlation_Selva.png", units = 'in',
       width = 7, height = 4, dpi = 300)




#Tabla de varianza cercana a cero
thresholder <- nearZeroVar(base_pe[,c(7:33)], saveMetrics = TRUE)
thresholder


# Eliminando variables
base_pe <- base_pe |>
  select(-Minería, -Coca_ha)


# Arbol de fuerza de correlación variables ------------------------------------------
r_d <- cor(base_pe[,c(8:31)], use="pairwise.complete.obs", method = "spearman")
r_d [is.na(r_d )] = 0
r_d.dist <- as.dist(1 - abs(r_d))
hc <- hclust(r_d.dist, method="complete")

labels(hc) <- c('Shannon index','Patch density','Dist. to roads',
                'Aggregation index', 'Infrastructure',
                'Agricultural and livestock','Agricultural workers',
                'Gross Domestic Product','Population',
                'Emigrants','Immigrants',
                'Rainfall','Dist.to timber concessions',
                'Dist. to ANP', 'Dist. to nat. commun.',
                'Yuca Root','River, lake or ocean',
                'Soil Moisture','Efficiency of public spending',
                'Poverty','IDH',
                'Dist. to mining cadastre', 'Temperature',
                'Dist. to rivers'
)
clust <- ggdendrogram(hc, rotate = F, size = 2, theme_dendro = FALSE) +
  scale_y_continuous(labels = c('1.00', '0.75', '0.50', '0.25', '0.00')) +
  geom_hline(yintercept = 0.25, colour = "red") +
  labs(x = 'Potential predictors', y = 'Strength') +
  ggtitle(label = 'Clustering of potential predictors based on the strength of association') +
  theme(plot.title = element_text(hjust = 0.5, size = 18),
        plot.subtitle = element_text(face = 'italic', hjust = 0.5, size = 13),
        axis.text.y = element_text(angle = 90, hjust = 0.5, size = 11),
        axis.text.x = element_text(size = 14, angle = 45, vjust = 1, hjust=1),
        axis.title.y = element_text(size = 16, family = "Lt"),
        axis.title.x = element_text(size = 16, family = "Lt"),
        legend.text = element_text(size = 9.5),
        plot.caption = element_text(hjust = 0, size = 8.5)) +
  theme(text = element_text(family = "Lt"))


clust

ggsave(plot = clust, filename = "mapas_graficos/clustering_potenc_predictors.png", units = 'in',
       width = 7, height = 4, dpi = 300)


##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Tabla con resultados de modelos ----
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#Modelo_3 sin causas directas
#Regiones
res_mod3_run1 <- readRDS("Modelos_2024/XGBOOST/output/mod_3/nr_run1.RDS")

imp_Sierra <- tibble(res_mod3_run1$SIERRA$model_var_importance_SIERRA) |>
  select(Variable = Feature, Value = Gain) |>
  mutate(Zone = 'Sierra')

test_Sierra <- res_mod3_run1$SIERRA$model_evaluation_SIERRA$metricas_test |>
  select(Variable = .metric, Value = .estimate) |>
  filter(Variable == 'rsq') |>
  mutate(Zone = 'Sierra')

values_Sierra <- rbind(imp_Sierra, test_Sierra)


imp_Selva <- tibble(res_mod3_run1$SELVA$model_var_importance_SELVA) |>
  select(Variable = Feature, Value = Gain) |>
  mutate(Zone = 'Selva')

test_Selva <- res_mod3_run1$SELVA$model_evaluation_SELVA$metricas_test |>
  select(Variable = .metric, Value = .estimate) |>
  filter(Variable == 'rsq') |>
  mutate(Zone = 'Selva')

values_Selva <- rbind(imp_Selva, test_Selva)

full_reg <- rbind(values_Sierra, values_Selva)

full_reg$Variable <- reorder(full_reg$Variable, full_reg$Value)


full_reg$Zone = as.factor(full_reg$Zone)




grap_mod3 <- ggplot(full_reg, aes(x = Zone, y = Value, fill = Variable)) +
  geom_bar(stat="identity", color="black",
           position = "dodge") +
  scale_fill_manual(values = c("#F0E442", "#E69F00", "#56B4E9", "#009E73",
                               "#D55E00","#0072B2",  "#CC79A7"),
                    labels=c('Dist. to mining cadastre','Dist. to ANP',
                             'Population','Aggregation index',
                             'Temperature','Rainfall',
                             expression("R"^2*" test"))) +
  theme(legend.position="bottom")+ guides(fill=guide_legend(title=" ")) +
  labs(x = ' ', y = expression("R"^2*"/"*"Importance of Predictors"*"  (%)")) +
  ggtitle(label = 'Model without proximate causes') +
  theme(plot.title = element_text(hjust = 0.5, size = 18),
        plot.subtitle = element_text(face = 'italic', hjust = 0.5, size = 13),
        axis.text.y = element_text(angle = 0, hjust = 0.5, size = 11),
        axis.text.x = element_text(size = 14, angle = 0, vjust = 1, hjust=0.5),
        axis.title.y = element_text(size = 14, family = "Lt"),
        axis.title.x = element_text(size = 14, family = "Lt"),
        legend.text = element_text(size = 14),
        plot.caption = element_text(hjust = 0, size = 8.5)) +
  theme(text = element_text(family = "Lt"))



ggsave(plot = grap_mod3, filename = "mapas_graficos/results_mod_und.png", units = 'in',
       width = 7, height = 4, dpi = 300)



#Modelo_4  incluye causas directas
#Regiones
res_mod4_run <- readRDS("Modelos_2024/XGBOOST/output/mod_4/nr_run1.RDS")

imp_Sierra_mod4 <- tibble(res_mod4_run$SIERRA$model_var_importance_SIERRA) |>
  select(Variable = Feature, Value = Gain) |>
  mutate(Zone = 'Sierra')

test_Sierra_mod4 <- res_mod4_run$SIERRA$model_evaluation_SIERRA$metricas_test |>
  select(Variable = .metric, Value = .estimate) |>
  filter(Variable == 'rsq') |>
  mutate(Zone = 'Sierra')

values_Sierra_mod4 <- rbind(imp_Sierra_mod4, test_Sierra_mod4)


imp_Selva_mod4 <- tibble(res_mod4_run$SELVA$model_var_importance_SELVA) |>
  select(Variable = Feature, Value = Gain) |>
  mutate(Zone = 'Selva')

test_Selva_mod4 <- res_mod4_run$SELVA$model_evaluation_SELVA$metricas_test |>
  select(Variable = .metric, Value = .estimate) |>
  filter(Variable == 'rsq') |>
  mutate(Zone = 'Selva')

values_Selva_mod4 <- rbind(imp_Selva_mod4, test_Selva_mod4)

full_reg_mod4 <- rbind(values_Sierra_mod4, values_Selva_mod4)

full_reg_mod4$Variable <- reorder(full_reg_mod4$Variable, full_reg_mod4$Value)


full_reg_mod4$Zone = as.factor(full_reg_mod4$Zone)


## Color / Category
# "#F0E442" ---- 'Dist. to mining cadastre'
# "#E69F00" ---- 'Dist. to ANP'
# "#56B4E9" ---- 'Population'
# "#009E73" ---- 'Aggregation index'
# "#D55E00" ---- 'Temperature'
# "#0072B2" ---- 'Rainfall'
# "#CC79A7" ---- "R2"
# "#B6BBC4" ---- "Infraestructure"
# "#FFE382" ---- 'Agricultural and livestock'


grap_mod4 <- ggplot(full_reg_mod4, aes(x = Zone, y = Value, fill = Variable)) +
  geom_bar(stat="identity", color="black",
           position = "dodge") +
  scale_fill_manual(values = c("#E69F00", "#D55E00", "#F0E442", "#56B4E9",
                               "#B6BBC4", "#009E73", "#0072B2", "#FFE382", "#CC79A7"),
                    labels=c('Dist. to ANP','Temperature','Dist. to mining cadastre',
                             'Population','Infraestructure',
                             'Aggregation index','Rainfall',
                             'Agricultural and livestock',
                             expression("R"^2*" test"))) +
  theme(legend.position="bottom")+ guides(fill=guide_legend(title=" ")) +
  labs(x = ' ', y = expression("R"^2*"/"*"Importance of Predictors"*"  (%)")) +
  ggtitle(label = 'Model with proximate causes') +
  theme(plot.title = element_text(hjust = 0.5, size = 18),
        plot.subtitle = element_text(face = 'italic', hjust = 0.5, size = 13),
        axis.text.y = element_text(angle = 0, hjust = 0.5, size = 11),
        axis.text.x = element_text(size = 14, angle = 0, vjust = 1, hjust=0.5),
        axis.title.y = element_text(size = 14, family = "Lt"),
        axis.title.x = element_text(size = 14, family = "Lt"),
        legend.text = element_text(size = 14),
        plot.caption = element_text(hjust = 0, size = 8.5)) +
  theme(text = element_text(family = "Lt"))



ggsave(plot = grap_mod4, filename = "mapas_graficos/results_mod_caus_prox.png", units = 'in',
       width = 7, height = 4, dpi = 300)
