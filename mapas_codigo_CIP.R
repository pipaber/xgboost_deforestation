#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#                                                                              #
#                 Código para mapas - estudio deforestación                    #
#                                                                              #
#             Consultora: Corina Navarrete                                     #
#            Institución: Centro Internacional de la Papa                      #
#                                                                              #
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Preparación ----
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++





# Load libraries ----------------------------------------------------------
require(pacman)
pacman::p_load(tidyverse, sf, ggspatial, hrbrthemes, ggthemes, ggrepel, 
               terra, RColorBrewer, geodata, rnaturalearthdata, rnaturalearth,
               ggpattern, showtext, xlsx, ggfx)

g <- gc(reset = T)
rm(list = ls())
options(scipen = 999)


font_add_google("Lato", "Lt")
showtext_auto()                                          



# Load data ---------------------------------------------------------------
distritos <- st_read("distritos_estudio.gpkg") |> 
  dplyr::select(ubigeo) |> 
  st_transform(crs = "EPSG:4326") 

amaz <- st_read("limiteamazonica.gpkg")
per <- st_read("pais.gpkg")
wrld <- ne_countries(returnclass = 'sf')
wrld <- filter(wrld, continent %in% c('South America'))
dep <- st_read("departamentos_Peru.gpkg")
dep_a <- st_centroid(dep)                                       
dep_l <- cbind(dep, st_coordinates(st_centroid(dep$geom))) 


# Study area ---------------------------------------------------------

# Main map
main_p <- ggplot() + 
  geom_sf(data = st_as_sf(distritos), fill = 'white', col = 'grey60', 
          aes(shape = "Study districts")) + #pattern_colour  = 'grey60',pattern_spacing = 0.05, 
  geom_sf(data = per, fill = NA, col = 'black', aes(cex = "Political boundary of Peru")) +
  geom_sf(data = amaz, fill = NA, col = 'green', aes(alpha = "Peruvian amazon border")) +
  labs(x = 'Longitude', y = 'Latitude', alpha = "", shape = "", cex = "") +
  coord_sf(xlim = ext(per)[1:2], ylim = ext(per)[3:4]) + 
  ggtitle(label = 'Study area') +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        plot.subtitle = element_text(face = 'italic', hjust = 0.5, size = 10),
        axis.text.y = element_text(angle = 90, hjust = 0.5, size = 9),
        axis.text.x = element_text(size = 9),
        axis.title.y = element_text(size = 12),
        axis.title.x = element_text(size = 12),
        legend.text = element_text(size = 11),
        plot.caption = element_text(hjust = 0, size = 8.5)) +
  theme(text = element_text(family = "Lt")) +
  annotation_scale(location = "bl") +
  annotation_north_arrow(location = "tr", which_north = "true",
                         height = unit(1, "cm"),
                         width = unit(1, "cm"),
                         pad_x = unit(-0.001, "in"), pad_y = unit(0.1, "in"), 
                         style = north_arrow_fancy_orienteering())


# Localization map

PE <- filter(wrld, name %in% c('Peru'))
extn <- ext(PE)


loc <- ggplot() +
  geom_sf(data = wrld, fill = 'grey60', col = 'grey40') +
  geom_sf(data = st_as_sf(PE), fill = 'darkred', col = 'grey40') + 
  coord_sf(xlim = c(-85, -60), ylim = c(-25, 0)) + 
  theme_bw() +
  theme(axis.text.y = element_blank(), 
        axis.text.x = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(), 
        panel.spacing = unit(c(0,0,0,0), "cm"),
        plot.margin = unit(c(0,0,0,0), "cm"),
        axis.title = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        legend.position = "none", 
        panel.border = element_rect( color = "grey20", fill = NA, size = 0.4),
        panel.background = element_rect(fill = "aliceblue")) +
  labs(x = '', y = '')


# Add the localization map to the main map
full <- main_p +
  annotation_custom(ggplotGrob(loc), xmin = -81.5, xmax = -78, ymin = -18, ymax = -13)



ggsave(plot = full, filename = 'mapas_graficos/study_area.png',
       units = 'in', width = 5, height = 5, dpi = 300)



# emerging hot spots map ----------------------------------------------
cl_hotspot <- st_read("ehsa.gpkg")|> 
  st_transform(crs = "EPSG:4326")  



cl_hotspot$group <- fct_recode(cl_hotspot$classification, L0 = 'persistent coldspot',
                               L0 = 'sporadic coldspot',L0 = 'consecutive coldspot', L0 = 'new coldspot',
                               L1 = 'no pattern detected', L2 = 'new hotspot', L3 = 'sporadic hotspot', 
                               L4 = 'consecutive hotspot', L5 = 'persistent hotspot')

levels(cl_hotspot$group) 
cl_hotspot$group <- factor(cl_hotspot$group, levels=c('L0', 'L1', 'L2', 'L3', 'L4', 'L5'))




EH_map <- ggplot() + 
  geom_sf(data = st_as_sf(cl_hotspot), col = 'white'
          , aes(fill = group)) + 
  scale_fill_manual(name="Categories of hot/cold spots",
                    values = c("#4242ff", "#c1c1c1", "#fddbc7", "#f4a582", "#d6604d", "#b2182b"),
                    labels = c("Cold spot category", "No pattern detected", "New hot spot", "Sporadic hot spot",
                               "Consecutive hot spot", "Persistent hot spot")) +
  geom_sf(data = dep, fill = NA, col = '#333333', ) +
  geom_sf(data = per, fill = NA, col = 'black') +
  geom_text_repel(data =dep_l, colour ='#000000',  aes(x=X, y=Y, label = NOMBDEP), size = 4, fontface = "bold",bg.color = "white",
                  bg.r = 0.25) +
  labs(x = 'Longitude', y = 'Latitude') + #The R package: {khroma}  BuRd
  coord_sf(xlim = ext(per)[1:2], ylim = ext(per)[3:4]) + 
  ggtitle(label = 'Spatiotemporal analysis of deforestation') +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        plot.subtitle = element_text(face = 'italic', hjust = 0.5, size = 10),
        axis.text.y = element_text(angle = 90, hjust = 0.5, size = 9),
        axis.text.x = element_text(size = 9),
        axis.title.y = element_text(size = 12),
        axis.title.x = element_text(size = 12),
        legend.text = element_text(size = 11),
        plot.caption = element_text(hjust = 0, size = 8.5)) +
  theme(text = element_text(family = "Lt")) + 
  annotation_scale(location = "bl") +
  annotation_north_arrow(location = "tr", which_north = "true",
                         height = unit(1, "cm"),
                         width = unit(1, "cm"),
                         pad_x = unit(-0.001, "in"), pad_y = unit(0.1, "in"), 
                         style = north_arrow_fancy_orienteering())



ggsave(plot = EH_map, filename = 'mapas_graficos/cluster.png',
       units = 'in', width = 5, height = 5, dpi = 300)

# Relationship between deforestation and the aggregation index  ------------------




cor_ind_ag_def <- st_read("defor_aggregation_index.gpkg")|> 
  st_transform(crs = "EPSG:4326")

cor_ind_ag_def <- cor_ind_ag_def |> 
  mutate(legend = case_when(
    cor <= -0.5 ~ "Strong negative",
    cor <= -0.3 ~ "Moderate negative",
    cor <= -0.1 ~ "Weak negative",
    cor < 0.1 ~ "No or negligible relationship",
    cor < 0.3 ~ "Weak positive",
    cor < 0.5 ~ "Moderate positive",
    TRUE ~ "Strong positive" 
  )) |> 
  mutate(legend = as_factor(legend)|> 
           fct_relevel("Strong negative", "Moderate negative", "Weak negative",
                       "No or negligible relationship", "Weak positive", "Moderate positive",
                       "Strong positive"))

coca <- st_read("dist_puntos_pres_coca.gpkg") |>
  filter(tot_ha > 0)

plm <- st_read("palma_2016_2020.gpkg")|> 
  st_transform(crs = "EPSG:4326")

agric <- st_read("cultivos_zona_estudio.gpkg")

mining <- st_read("mineria_ibc_2020.gpkg")




## deforestation and the aggregation index ----
map_cor_0 <- ggplot(cor_ind_ag_def) + 
  geom_sf(col = 'white', aes(fill = legend)) +
  scale_fill_manual(name="Spearman Correlation",
                    values = c( "#b35806", "#e08214","#fdb863","#d8daeb","#b2abd2","#8073ac", "#542788"))+
  geom_sf(data = dep, fill = NA, col = '#333333') +
  geom_sf(data = per, fill = NA, col = 'black') +
  geom_text_repel(data =dep_l, colour ='#000000',  aes(x=X, y=Y, label = NOMBDEP), size = 4, fontface = "bold",bg.color = "white",
                  bg.r = 0.25) +
  labs(x = 'Longitude', y = 'Latitude') +
  coord_sf(xlim = ext(per)[1:2], ylim = ext(per)[3:4]) + 
  ggtitle(label = 'Relationship between deforestation and the aggregation index',
          subtitle = 'Time-based') +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        plot.subtitle = element_text(face = 'italic', hjust = 0.5, size = 13),
        axis.text.y = element_text(angle = 90, hjust = 0.5, size = 9),
        axis.text.x = element_text(size = 9),
        axis.title.y = element_text(size = 12),
        axis.title.x = element_text(size = 12),
        legend.text = element_text(size = 11),
        plot.caption = element_text(hjust = 0, size = 8.5)) +
  theme(text = element_text(family = "Lt")) + 
  annotation_scale(location = "bl") +
  annotation_north_arrow(location = "tr", which_north = "true",
                         height = unit(1, "cm"),
                         width = unit(1, "cm"),
                         pad_x = unit(-0.001, "in"), pad_y = unit(0.1, "in"), 
                         style = north_arrow_fancy_orienteering())
map_cor_0


ggsave(plot = map_cor_0, filename = 'mapas_graficos/ind_agreg_vs_defor.png', 
       units = 'in', width = 5, height = 5, dpi = 300)



## illegal mining ----

map_cor_1 <- ggplot(cor_ind_ag_def) + 
  geom_sf(col = 'white', aes(fill = legend)) +
  scale_fill_manual(name="Spearman Correlation",
                    values = c( "#b35806", "#e08214","#fdb863","#d8daeb","#b2abd2","#8073ac", "#542788"))+
  geom_sf_pattern(data = mining,  col = '#ffff72', fill = NA, 
                  pattern_colour  = '#ffff72',pattern_spacing = 0.01, aes(shape = "Illegal mining")) +
  geom_sf(data = dep, fill = NA, col = '#333333') +
  geom_sf(data = per, fill = NA, col = 'black') +
  geom_text_repel(data =dep_l, colour ='#000000',  aes(x=X, y=Y, label = NOMBDEP), size = 4, fontface = "bold",bg.color = "white",
                  bg.r = 0.25) +
  labs(x = 'Longitude', y = 'Latitude',  shape = '') +
  coord_sf(xlim = ext(per)[1:2], ylim = ext(per)[3:4]) + 
  ggtitle(label = 'Overlapping with illegal mining') +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        plot.subtitle = element_text(face = 'italic', hjust = 0.5, size = 13),
        axis.text.y = element_text(angle = 90, hjust = 0.5, size = 9),
        axis.text.x = element_text(size = 9),
        axis.title.y = element_text(size = 12),
        axis.title.x = element_text(size = 12),
        legend.text = element_text(size = 11),
        plot.caption = element_text(hjust = 0, size = 8.5)) +
  theme(text = element_text(family = "Lt")) + 
  annotation_scale(location = "bl") +
  annotation_north_arrow(location = "tr", which_north = "true",
                         height = unit(1, "cm"),
                         width = unit(1, "cm"),
                         pad_x = unit(-0.001, "in"), pad_y = unit(0.1, "in"), 
                         style = north_arrow_fancy_orienteering())


map_cor_1


ggsave(plot = map_cor_1, filename = 'mapas_graficos/ind_agreg_vs_defor_mining.png', 
       units = 'in', width = 5, height = 5, dpi = 300)



## agricultural land ----
map_cor_2 <- ggplot(cor_ind_ag_def) + 
  geom_sf(col = 'white', aes(fill = legend)) +
  scale_fill_manual(name="Spearman Correlation",
                    values = c( "#b35806", "#e08214","#fdb863","#d8daeb","#b2abd2","#8073ac", "#542788"))+
  geom_sf(data = agric,  col = NA, fill = '#508104', alpha = 0.7, aes(shape = "Agricultural land")) +
  geom_sf(data = dep, fill = NA, col = '#333333') +
  geom_sf(data = per, fill = NA, col = 'black') +
  geom_text_repel(data =dep_l, colour ='#000000',  aes(x=X, y=Y, label = NOMBDEP), size = 4, fontface = "bold",bg.color = "white",
                  bg.r = 0.25) +
  labs(x = 'Longitude', y = 'Latitude',  shape = '') +
  coord_sf(xlim = ext(per)[1:2], ylim = ext(per)[3:4]) + 
  ggtitle(label = 'Overlapping with agricultural land') +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        plot.subtitle = element_text(face = 'italic', hjust = 0.5, size = 10),
        axis.text.y = element_text(angle = 90, hjust = 0.5, size = 9),
        axis.text.x = element_text(size = 9),
        axis.title.y = element_text(size = 12),
        axis.title.x = element_text(size = 12),
        legend.text = element_text(size = 11),
        plot.caption = element_text(hjust = 0, size = 8.5)) +
  theme(text = element_text(family = "Lt")) + 
  annotation_scale(location = "bl") +
  annotation_north_arrow(location = "tr", which_north = "true",
                         height = unit(1, "cm"),
                         width = unit(1, "cm"),
                         pad_x = unit(-0.001, "in"), pad_y = unit(0.1, "in"), 
                         style = north_arrow_fancy_orienteering())



map_cor_2

ggsave(plot = map_cor_2, filename = 'mapas_graficos/agric.png', 
       units = 'in', width = 5, height = 5, dpi = 300)


## oil palm ----

map_cor_3 <- ggplot(cor_ind_ag_def) + 
  geom_sf(col = 'white', aes(fill = legend)) +
  scale_fill_manual(name="Spearman Correlation",
                    values = c( "#b35806", "#e08214","#fdb863","#d8daeb","#b2abd2","#8073ac", "#542788"))+
  geom_sf(data = plm, col = 'red', fill = 'red', alpha = 0.9, linewidth= 0.05, aes(shape = "Palm oil")) +
  geom_sf(data = dep, fill = NA, col = '#333333') +
  geom_sf(data = per, fill = NA, col = 'black') +
  geom_text_repel(data =dep_l, colour ='#000000',  aes(x=X, y=Y, label = NOMBDEP), size = 4, fontface = "bold",bg.color = "white",
                  bg.r = 0.25) +
  labs(x = 'Longitude', y = 'Latitude',  shape = '') +
  coord_sf(xlim = ext(per)[1:2], ylim = ext(per)[3:4]) + 
  ggtitle(label = 'Overlapping with palm oil') +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        plot.subtitle = element_text(face = 'italic', hjust = 0.5, size = 10),
        axis.text.y = element_text(angle = 90, hjust = 0.5, size = 9),
        axis.text.x = element_text(size = 9),
        axis.title.y = element_text(size = 12),
        axis.title.x = element_text(size = 12),
        legend.text = element_text(size = 11),
        plot.caption = element_text(hjust = 0, size = 8.5)) +
  theme(text = element_text(family = "Lt")) + 
  annotation_scale(location = "bl") +
  annotation_north_arrow(location = "tr", which_north = "true",
                         height = unit(1, "cm"),
                         width = unit(1, "cm"),
                         pad_x = unit(-0.001, "in"), pad_y = unit(0.1, "in"), 
                         style = north_arrow_fancy_orienteering())

map_cor_3

ggsave(plot = map_cor_3, filename = 'mapas_graficos/palma_aceitera.png', 
       units = 'in', width = 5, height = 5, dpi = 300)




## Mapa indice de agregacion/deforestacion c/ coca ----
coca_rec <- coca|> 
  mutate(tmn_coca = case_when(
    tot_ha <= 5000 ~ "0 - 5000 ha",
    tot_ha <= 15000 ~ "5000 - 15000 ha",
    tot_ha <= 25000 ~ "15000 - 25000 ha",
    tot_ha <= 35000 ~ "25000 - 35000 ha",
  )) |> 
  mutate(tmn_coca = as_factor(tmn_coca)|> 
           fct_relevel("0 - 5000 ha", "5000 - 15000 ha", 
                       "15000 - 25000 ha", "25000 - 35000 ha"))


map_cor_4 <- ggplot(cor_ind_ag_def) + 
  geom_sf(col = 'white', aes(fill = legend)) +
  scale_fill_manual(name="Spearman Correlation",
                    values = c( "#b35806", "#e08214","#fdb863","#d8daeb","#b2abd2","#8073ac", "#542788"))+
  geom_sf(data = coca_rec, fill = 'green', col = 'green', aes(size = tmn_coca)) +
  scale_size_discrete(range = c(0.1, 1)) + 
  geom_sf(data = dep, fill = NA, col = '#333333') +
  geom_sf(data = per, fill = NA, col = 'black') +
  geom_text_repel(data =dep_l, colour ='#000000',  aes(x=X, y=Y, label = NOMBDEP), size = 4, fontface = "bold",bg.color = "white",
                  bg.r = 0.25) +
  labs(x = 'Longitude', y = 'Latitude', size ='Coca Crop') +
  coord_sf(xlim = ext(per)[1:2], ylim = ext(per)[3:4]) + 
  ggtitle(label = 'Overlapping with coca crop') +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        plot.subtitle = element_text(face = 'italic', hjust = 0.5, size = 10),
        axis.text.y = element_text(angle = 90, hjust = 0.5, size = 9),
        axis.text.x = element_text(size = 9),
        axis.title.y = element_text(size = 12),
        axis.title.x = element_text(size = 12),
        legend.text = element_text(size = 11),
        plot.caption = element_text(hjust = 0, size = 8.5)) +
  theme(text = element_text(family = "Lt")) + 
  annotation_scale(location = "bl") +
  annotation_north_arrow(location = "tr", which_north = "true",
                         height = unit(1, "cm"),
                         width = unit(1, "cm"),
                         pad_x = unit(-0.001, "in"), pad_y = unit(0.1, "in"), 
                         style = north_arrow_fancy_orienteering())

map_cor_4

ggsave(plot = map_cor_4, filename = 'mapas_graficos/coca.png', 
       units = 'in', width = 5, height = 5, dpi = 300)


library(patchwork)

mp <- ((map_cor_1 + map_cor_2)/(map_cor_3 + map_cor_4)) +
  plot_layout(guides = "collect")
ggsave(plot = mp, filename = 'mapas_graficos/mapas_conjunto.png', 
       units = 'in', width = 8, height = 8, dpi = 300)
