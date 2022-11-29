
library(seewave)
library(tuneR)
library(soundecology)

#---Setting the working directory-----------------------------------------------
a <-setwd("C:/Users/Santigo Ruiz/Burned/")

#---File list with the wav files inside the working directory----------------
files_sound <- list.files(a,".wav")

#---Dataframe where the output will be stored------------------
data <- as.data.frame(files_sound)

#--We will make a empty column per each index we want to calculate--------------

data$H=NA
data$Hf=NA
data$ACI=NA
data$ACI1=NA
data$ACI2=NA
data$ACI3=NA
data$ACI4=NA
data$ACI5=NA
data$ADI=NA
data$AEI=NA
data$BI=NA


####data$nombres=files_sound

# Loop for the acoustic indices calculation-----

for(i in 1:length(files_sound))

{
song_name <- files_sound[i]
song <- readWave(song_name)
spec <- meanspec(song, 
                 f=22050, 
                 plot=FALSE)

ent=H(song,
      f=22050)

timent=sh(spec)

AcCI=ACI(song)
AcCI1=ACI(song,flim=c(0,2)) # Acoustic Complexity index for different frequency bandwidth
AcCI2=ACI(song,flim=c(2,4))
AcCI3=ACI(song,flim=c(4,6))
AcCI4=ACI(song,flim=c(6,8))
AcCI5=ACI(song,flim=c(8,10))

adi=acoustic_diversity(song, 
                       max_freq = 10000, 
                       db_threshold = -50, 
                       freq_step = 1000, 
                       shannon = TRUE)

aev=acoustic_evenness(song, 
                      max_freq = 10000, 
                      db_threshold = -50, 
                      freq_step = 1000)

bi=bioacoustic_index(song, 
                     min_freq = 0, 
                     max_freq = 10000, 
                     fft_w = 512)

# Then, we will append the loop output inside the empty columns-----------

data$H[i]=ent
data$Hf[i]=timent
data$ACI[i]=AcCI
data$ACI1[i]=AcCI1
data$ACI2[i]=AcCI2
data$ACI3[i]=AcCI3
data$ACI4[i]=AcCI4
data$ACI5[i]=AcCI5
data$ADI[i]=adi$adi_left
data$AEI[i]=aev$aei_left
data$BI[i]=bi$left_area
}

# Lastly, we can write the dataframe as .csv


write.table(data,"C:/Users/Santigo Ruiz/Desktop/CANTOS/TESIS_2.0/Result_QUEMADOS.csv")



















# Para cambiar el formato de una tabla:
x <- read.table("C:/Users/Santigo Ruiz/Desktop/CANTOS/TESIS_2.0/Result_Ejemplo.csv")
write.table(x,"C:/Users/Santigo Ruiz/Desktop/CANTOS/TESIS_2.0/Result_Ejemplo.txt")

x <- read.table("C:/Users/Santigo Ruiz/Desktop/CANTOS/TESIS/Result_Index/Res/Result_Negro8.csv")
write.table(x,"C:/Users/Santigo Ruiz/Desktop/CANTOS/TESIS/Resultado_Negro8.txt")


#----------------------------CREACION DE BOXPLOT-------------------------------------------------------------------

# Paquete necesario
library(graphics)

# Se crea un objeto por cada uno de los archivos (indices por grabadora) que tienen los resultados a graficar
# El argumento "head=T" significa que la primera fila son los nombres de las columnas

x <- read.table("d:/Users/Santiago Ruiz/Desktop/CANTOS/TESIS/Datos_Indices/Quema-5D.txt",
                head=T,
                sep = "",
                quote = "/",
                dec = ".")
x$tipo <- "quemado"
x$sitio <- "quemado5"

y <- read.table("d:/Users/Santiago Ruiz/Desktop/CANTOS/TESIS/Datos_Indices/Quema-8D.txt",
                head=T,
                sep = "",
                quote = "",
                dec = ".")
y$tipo <- "quemado"
y$sitio <- "quemado8"

z <- read.table("d:/Users/Santiago Ruiz/Desktop/CANTOS/TESIS/Datos_Indices/Quema-10D.txt",
                head=T,
                sep = "",
                quote = "/",
                dec = ".")
z$tipo <- "quemado"
z$sitio <- "quemado10"

a <- read.table("d:/Users/Santiago Ruiz/Desktop/CANTOS/TESIS/Datos_Indices/Bosque-2D.txt",
                head=T,
                sep = "",
                quote = "/",
                dec = ".")
a$tipo <- "bosque"
a$sitio <- "bosque2"

b <- read.table("d:/Users/Santiago Ruiz/Desktop/CANTOS/TESIS/Datos_Indices/Bosque-3D.txt",
                head=T,
                sep = "",
                quote = "/",
                dec = ".")
b$tipo <- "bosque"
b$sitio <- "bosque3"

c <- read.table("d:/Users/Santiago Ruiz/Desktop/CANTOS/TESIS/Datos_Indices/Bosque-4D.txt",
                head=T,
                sep = "",
                quote = "/",
                dec = ".")
c$tipo <- "bosque"
c$sitio <- "bosque4"


# Se crea un objeto tabla con la función "rbind" con cada uno de los objetos creados anteriormente
tabla <- rbind (a,b,c,x,y,z)

bosque <- rbind (a,b,c)
Quemado <- rbind (x,y,z)



ggplot(bosque, aes(x = X.Hora., y = X.ACI1.)) +
  geom_boxplot()




  boxplot(data$ACI)
write.table(tabla, "C:/Users/Santigo Ruiz/Desktop/CANTOS/TESIS/Result_Index/Consolidado2.txt")


#fin
