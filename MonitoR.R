
#------------------------Required packages--------------------------------------------------
library(tuneR)
library(monitoR)
library(seewave)
library(RODBC)

  work <- setwd("C:/Users/Santigo Ruiz/Desktop/CANTOS/TESIS/Dia-Noche/NOCHE/Negro5-N")
  files_sound <- list.files(work,".wav")
  dat <- as.data.frame(files_sound)
  
  dat$template=NA
  dat$date.time=NA
  dat$time=NA
  dat$score=NA
  dat$detection=NA

#----------------------Templates creation-------------------------------------------
  x <- readWave("C:/Users/Santigo Ruiz/Desktop/CANTOS/TESIS/Sp_candidatas/Templates/Choliba/Choliba1.wav",
                from = 1, to = Inf, 
                units = "seconds" )
  
  w <- readWave("C:/Users/Santigo Ruiz/Desktop/CANTOS/TESIS/Sp_candidatas/Templates/Choliba/Choliba2.wav",
                from = 1, to = Inf, 
                units = "seconds" )



#------------------------------BINARY POINT MATCHING----------------------------------
work <- setwd("C:/Users/Santigo Ruiz/Desktop/CANTOS/TESIS/Dia-Noche/NOCHE/Negro5-N")
files_sound <- list.files(work,".wav")
tabla <- as.data.frame(files_sound)
tabla$template=NA
tabla$date.time=NA
tabla$time=NA
tabla$score=NA



wbt1 <- makeBinTemplate("C:/Users/Santigo Ruiz/Desktop/CANTOS/TESIS/Sp_candidatas/Templates/Choliba/Choliba1.wav",
                        t.lim = c(0.5,2),
                        frq.lim = c(0.4,2.5),
                        amp.cutoff = -30,
                        buffer = 0.5, 
                        name = "w1")

wbt2 <- makeBinTemplate("C:/Users/Santigo Ruiz/Desktop/CANTOS/TESIS/Sp_candidatas/Templates/Choliba/Choliba2.wav",
                        t.lim = c(0.5,1.9),
                        frq.lim = c(0.4,4),
                        amp.cutoff = -40,
                        buffer = 1, 
                        name = "w2")

#--------------------------Templates combination-----------------------------------------
btemps <- combineBinTemplates(wbt1, wbt2)


for(i in 1:length(files_sound))
{
  
  song_name <- files_sound[i]
  song <- readWave(song_name,
                  header = FALSE,
                  from = 1,to = Inf, 
                  units = "seconds")
  
bscores <- binMatch(song_name,
                    btemps,
                    rec.tz = "UTC",
                    time.source = "fileinfo")



#------------------------------Detection-----------------------------------
bdetects <- findPeaks(bscores)



templateCutoff(bdetects) <- c(w2 = 2, default = 2)


# The detections are:
detecciones<- getDetections(bdetects)
tabla$template[i]= detecciones$template
tabla$date.time[i]= detecciones$date.time
tabla$time[i]=detecciones$time
tabla$score[i]=detecciones$score

write.table(tabla,"C:/Users/Santigo Ruiz/Desktop/CANTOS/TESIS/Sp_candidatas/Templates/Choliba/Choliba2.txt")
}

final <-getDetections(cdetects, output = "list")

write.table(detecciones,"C:/Users/Santigo Ruiz/Desktop/CANTOS/TESIS/Sp_candidatas/Templates/Choliba/Choliba1.1.txt")
