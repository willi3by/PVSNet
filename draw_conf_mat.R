draw_confusion_matrix <- function(cmtrx) {
  
  total <- sum(cmtrx$table)
  
  res <- as.numeric(cmtrx$table)
  
  # Generate color gradients. Palettes come from RColorBrewer.
  
  greenPalette <- c("#F7FCF5","#E5F5E0","#C7E9C0","#A1D99B","#74C476","#41AB5D","#238B45","#006D2C","#00441B")
  
  redPalette <- c("#FFF5F0","#FEE0D2","#FCBBA1","#FC9272","#FB6A4A","#EF3B2C","#CB181D","#A50F15","#67000D")
  
  getColor <- function (greenOrRed = "green", amount = 0) {
    
    if (amount == 0)
      
      return("#FFFFFF")
    
    palette <- greenPalette
    
    if (greenOrRed == "red")
      
      palette <- redPalette
    
    colorRampPalette(palette)(100)[10 + ceiling(90 * amount / total)]
    
  }
  
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n', bty='n')
  
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix
  
  classes = colnames(cmtrx$table)
  
  rect(150, 430, 240, 370, col=getColor("green", res[1]))
  
  text(195, 440, classes[1], cex=1.2) 
  
  rect(250, 430, 340, 370, col=getColor("red", res[3]))
  
  text(295, 440, classes[2], cex=1.2) 
  
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  
  text(245, 450, 'Actual', cex=1.3, font=2)
  
  rect(150, 305, 240, 365, col=getColor("red", res[2]))
  
  rect(250, 305, 340, 365, col=getColor("green", res[4]))
  
  text(140, 400, classes[1], cex=1.2, srt=90)
  
  text(140, 335, classes[2], cex=1.2, srt=90)
  
  # add in the cmtrx results

  text(195, 400, res[1], cex=1.6, font=2, col='white')

  text(195, 335, res[2], cex=1.6, font=2, col='white')

  text(295, 400, res[3], cex=1.6, font=2, col='white')

  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
}
