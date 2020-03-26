#Fonction à charger impérativement

f_tilde1 <- function(x, y){  #f_tilde1, n'est pas une densité !!
  return(exp(-0.5 * (0.25 * x^2 + y^2)) * (abs(y) <= 1))  #représente l'indicatrice, c'est un booleen
}

f_tilde2 <- function(x,y){  # pas une densité
  return((cos(x)^2) + 0.5 * (sin(3*y)^2) * (cos(x)^4) * exp(-0.5 * (0.25 * x^2 + y^2)))
}

f_tilde3 <- function(x, y){  #f_tilde1, n'est pas une densité !!
  return(exp(-0.5 * (0.25 * x^2 + y^2)) * ((abs(y) <= 1) + 0.5))  #représente l'indicatrice, c'est un booleen
}

method_reject_opti_f1 <- function(n){ #algo renvoie n realisation de densité f    //la realisation est acceptee en moyenne au temps c * m1 = 1/(pnorm(1)-pnorm(-1))
  ans <- c()       #ce que l'on va retourner, vide pour l'instant  //
  c <- 1 / ((pnorm(1) - pnorm(-1)) * 4 * pi)
  m1 <- 4 * pi 
  compteur <- n   # nombre de réalisations, tant qu'on n'a pas atteint n
  # on poursuit l'algo tant qu'on n'a aps atteint n
  
  while(compteur > 0){
    # en moyenne il faut c * m1 * n tour de boucle pour avoir n réalisations
    
    unif <- runif(floor(compteur * c * m1 + 1)) #les uniformes
    
    y <- matrix(rnorm(floor(compteur * c * m1 + 1) * 2, mean = 0, sd = 1), ncol = 2)  #génération de n = compteur variables (de R2) suivant la loi g 
    #on génère n = compteur*m1 ! couples (X1,X2) de N(0.1)
    #y suivent la loi de g
    y[,1] <- y[,1] * 2         #on ajuste la variance des X1 pour obtenir une N(0,4)
    
    #on applique le test de rejet ou non
    
    z <- ((m1 * unif * dnorm(y[,1], mean = 0, sd = 2) * dnorm(y[,2])) <= (f_tilde1(y[,1], y[,2]))) * y 
    # Z est un booléen (0 ou 1) * la fonction f !! car le Temps T coincide
    #en fait on garde les y qui vérifient le test, et ils suivent une loi suivant f !
    #test de la forme  m1*g*U <= f
    ans <- rbind(ans,z[(z[,1] != 0 | z[,2] != 0),]) #en faisant ça je voulais éviter le cas (improbable) où l'on ôte une variable alors qu'elle n'a pas été rejetée , parceque X1 ou X2 vaut 0
    
    if (is.null(dim(z[(z[,1] != 0 | z[,2] != 0),]))){  #réglages techniques, sinon dans le else on pourrait avoir: compteur - NULL (à cause de la fct dim) qui implique une erreur sur le test while !
      compteur <- compteur - 1
    }else{
      compteur <- (compteur - dim(z[(z[,1] != 0 | z[,2] != 0),])[1])            #rbind() == concatenation verticale
    }
  }
  return(head(ans,n)) #extraction des n first row of the matrix
}

method_reject_opti_f2 <- function(n){ #algo renvoie n realisation de densité f    //la realisation est acceptee en moyenne au temps c2 * m2; c2 = 1/integral, malheureusement on ne sait l'estimer pr l'instant
  ans <- c()       #ce que l'on va retourner, vide pour l'instant  //
  c <- 0.1
  m1 <- 6 * pi 
  compteur <- n   # nombre de réalisations, tant qu'on n'a pas atteint n
  # on poursuit l'algo tant qu'on n'a aps atteint n
  
  while(compteur > 0){
    # en moyenne il faut c * m1 * n tour de boucle pour avoir n réalisations
    
    unif <- runif(floor(compteur * c * m1 + 1)) #les uniformes
    
    y <- matrix(rnorm(floor(compteur * c * m1 + 1) * 2, mean = 0, sd = 1), ncol = 2)  #génération de n = compteur variables (de R2) suivant la loi g 
    #on génère n = compteur*m1 ! couples (X1,X2) de N(0.1)
    #y suivent la loi de g
    y[,1] <- y[,1] * 2         #on ajuste la variance des X1 pour obtenir une N(0,4)
    
    #on applique le test de rejet ou non
    
    z <- ((m1 * unif * dnorm(y[,1], mean = 0, sd = 2) * dnorm(y[,2])) <= (f_tilde2(y[,1], y[,2]))) * y 
    # Z est un booléen (0 ou 1) * la fonction f !! car le Temps T coincide
    #en fait on garde les y qui vérifient le test, et ils suivent une loi suivant f !
    #test de la forme  m1*g*U <= f
    
    ans <- rbind(ans,z[(z[,1] != 0 | z[,2] != 0),]) #en faisant ça je voulais éviter le cas (improbable) où l'on ôte une variable alors qu'elle n'a pas été rejetée , parceque X1 ou X2 vaut 0
    
    if (is.null(dim(z[(z[,1] != 0 | z[,2] != 0),]))){  #cf version f1
      compteur <- compteur - 1
    }
    else{
      compteur <- (compteur - dim(z[(z[,1] != 0 | z[,2] != 0),])[1])            #rbind() == concatenation verticale
    }
  }
  return(head(ans,n)) #extraction des n first row of the matrix
}


#################### PARTIE 1: Algorithme du rejet ####################



second_marginal_function <- function(x){  #densite de la seconde marginale
  return((exp(-0.5 * x^2)/((pnorm(1) - pnorm(-1)) * sqrt(2 * pi)) )* (abs(x) <= 1))
}





## PARAMETRAGE TAILLE DU SAMPLE X = (X1,X2) ~ f_1

sample_size_n = 100000

f1_sample <- method_reject_opti_f1(sample_size_n)  

## PARAMETRAGE TAILLE DU SAMPLE X = (X1,X2) ~ f_1


       #   par(mfrow=(c(1,2))) #à utiliser si vous le desirez dim(z[z[,1]!=0,])



# Comparaison de la première marginale avec la première marginale théorique N(0,4)

hist(f1_sample[,1], breaks = 100, freq = F, col ='lavenderblush', main = "Histogram of 1st marginal of X ~ f1", ylab = "Probability density", xlim=c(-8,8))

lines(seq(-8,8,0.1),dnorm(seq(-8,8,0.1), mean = 0, sd = 2), col = "skyblue", lwd = 3)

legend("topleft", expression(paste('density of ', 'X'[1] )), col = "skyblue", lwd = 3, pt.cex = 1, cex = 0.8, bty = 'n')




# Comparaison de la seconde marginale avec la seconde marginale théorique

          par(mfrow=c(1,1))

hist(f1_sample[,2], breaks = 100, freq = F, col ='lavenderblush', main = "Histogram of 2nd marginal of X ~ f1", ylab = "Probability density")

lines(seq(-1,1,0.05),second_marginal_function(seq(-1,1,0.05)), col = "skyblue", lwd = 3)

legend("topleft", expression(paste('density of ', 'X'[2] )), col = "skyblue", lwd = 3, pt.cex = 1, cex = 0.8, bty = 'n')














#################### PARTIE 2: Retour sur les méthodes de réduction de variance ####################





## PARAMETRAGE TAILLE DU SAMPLE X = (X1,X2) ~ f_1

sample_size_n = 5000     # attention, ici l'algo d'Allocation proportionnelle prends du temps ! (sauvegarde de chaque étape n)  (evol_estim_AP)

f1_sample <- method_reject_opti_f1(sample_size_n)  

## PARAMETRAGE TAILLE DU SAMPLE X = (X1,X2) ~ f_1



### Premier cas


## Algorithmes d'étude de l'évolution de l'estimation selon n

evol_estim <- function(x){
  return(cumsum(x)/(1:length(x)))  #renvoie un vecteur de taille length(x)
}

evol_IC <- function(x, esperance, level = 0.95){   #retourne une data frame, esperance a le role d'un vect d'esperance
  n <- length(x)                
  s2 <- (cumsum(x^2) - (1:n) * esperance^2)/(0:(n-1)) # attention, cet estimateur n'est def que pour n >= 2 !
  bound.IC <- qnorm(0.5 * (level + 1)) * sqrt(s2/(1:n))  # q_1-alpha/2 * sqrt(varchap/n)
  return(data.frame(var = s2, binf = esperance - bound.IC, bsup = esperance + bound.IC))
}



## Méthode de Monte-Carlo classique (MC)

indicatrices <- (exp(f1_sample[,1]) + exp(f1_sample[,2])) >= 5
p_hat_MC <- evol_estim(indicatrices)
IC_MC <- evol_IC(indicatrices, p_hat_MC)  # par defaut alpha = 5%

    #Estimation Monte-Carlo classique finale de la proba p; variance et IC associés
    p_hat_MC[sample_size_n]
    IC_MC[sample_size_n,]
    
    #Plot de l'évolution en fonction de n
    plot(1:sample_size_n, p_hat_MC, ylim = c(0,1), type = "l", lwd = 1, col = 'skyblue', main = "Evolution de l'estimation classique de p")
    lines(2:sample_size_n, IC_MC$binf[2:sample_size_n], col = "firebrick", lwd = 1)  #plot des bornes de l'IC !
    lines(2:sample_size_n, IC_MC$bsup[2:sample_size_n], col = "firebrick", lwd = 1)  
    legend("topright", c("IC à 'level' %","estimation de p"), col = c("firebrick","skyblue"), lwd = c(2,1), pt.cex = 1, cex = 0.8, bty ="n")
    
    
    
    
## Méthode des variables antithétiques (VA)
    
y <- 0.5 * (((exp(f1_sample[,1]) + exp(f1_sample[,2])) >= 5) + ((exp(-1 * f1_sample[,1]) + exp(-1 * f1_sample[,2])) >= 5))                 #en fait on créé artificiellement de la nouvelle data (par transformation A)
p_hat_VA <- evol_estim(y)
IC_VA <- evol_IC(y, p_hat_VA)

    #Estimation finale,  de la proba p; variance et IC associés
    cov(((exp(f1_sample[,1]) + exp(f1_sample[,2])) >= 5), ((exp(-f1_sample[,1]) + exp(-f1_sample[,2])) >= 5))
    p_hat_VA[sample_size_n]
    IC_VA[sample_size_n,]
    
    #Plot de l'évolution en fonction de n
    plot(1:sample_size_n, p_hat_VA, ylim = c(0,1), type = "l", lwd = 1, col = 'skyblue', main = "Evolution de l'estimation VA de p")
    lines(2:sample_size_n, IC_VA$binf[2:sample_size_n], col = "firebrick", lwd = 1)  #plot des bornes de l'IC !
    lines(2:sample_size_n, IC_VA$bsup[2:sample_size_n], col = "firebrick", lwd = 1)  
    legend("topright", c("IC à 'level' %","estimation de p"), col = c("firebrick","skyblue"), lwd = c(2,1), pt.cex = 1, cex = 0.8, bty ="n")
    

    
    
    
## Méthode de la variable de contrôle (VC)
    
    h_0 <- function(x,y){  #indicatrice
      return(exp(x+y) >= 5)
    }
    
    h_0_bis <- function(x,y){  #indicatrice
      return(exp((x+y)*0.5) >= 5)
    }
    
    h_1 <- function(x,y){  #indicatrice   
      return((exp((x+y)) + 1) >= 5)
    }
    
    
    expct_h_0 <- function(n){    #estimation de E{h_0} par l'astuce des loi normales
      x <- rnorm(n, mean = 0, sd = 2)
      y <- rnorm(n)             #on obtient deux vecteurs de taille n
      
      return( mean( (1/(pnorm(1) - pnorm(-1))) * (exp(x + y) >= 5) * (abs(y) <= 1)))
    }
    
    control_term <- function(x, y, b, n_expct_h0){  #n_exp_h0 = nb de simulations pr approximer E{h0}
      return(b * (h_0(x,y) - expct_h_0(n_expct_h0)))
    }
    

    
    #on propose quelques valeurs de corr(h0,g); pour confirmer la proximité de notre fonction h0 avec h !
        
cor( x = ((exp(f1_sample[,1]) + exp(f1_sample[,2])) >= 5), y = h_0(f1_sample[,1], f1_sample[,2]))
    

cor( x = ((exp(f1_sample[,1]) + exp(f1_sample[,2])) >= 5), y = h_1(f1_sample[,1], f1_sample[,2]))
    
    
    
                  # g                            -  b*(h0 + E [h0]  )
        y_VC <- ((exp(f1_sample[,1]) + exp(f1_sample[,2])) >= 5) - control_term(f1_sample[,1], f1_sample[,2], b = 1, n_expct_h0 = 100000)            #g  on obtien un sample
        p_hat_VC <- evol_estim(y_VC)
        IC_VC <- evol_IC(y_VC, p_hat_VC)





    #Estimation finale,  de la proba p; variance et IC associés
    p_hat_VC[sample_size_n]
    IC_VC[sample_size_n,]
    
    
    #Plot de l'évolution en fonction de n
    plot(1:sample_size_n, p_hat_VC, ylim = c(0,1), type = "l", lwd = 1, col = 'skyblue', main = "Evolution de l'estimation VC de p")
    lines(2:sample_size_n, IC_VC$binf[2:sample_size_n], col = "firebrick", lwd = 1)  #plot des bornes de l'IC !
    lines(2:sample_size_n, IC_VC$bsup[2:sample_size_n], col = "firebrick", lwd = 1)  
    legend("topright", c("IC à 'level' %","estimation de p"), col = c("firebrick","skyblue"), lwd = c(2,1), pt.cex = 1, cex = 0.8, bty ="n")
    
    
    #Optimisons notre variable de contrôle b
    
b_optim <- function(x, y){  # return cov(g,h_0)/var(h_0)
  
  return( cov(((exp(x) + exp(y)) >= 5), h_0(x,y)) / var(h_0(x,y)))
  
}

    #Nouvelle estimation de p_hat

y_VC <- ((exp(f1_sample[,1]) + exp(f1_sample[,2])) >= 5) - control_term(f1_sample[,1], f1_sample[,2], b = b_optim(x = f1_sample[,1], y = f1_sample[,2]), n_expct_h0 = 100000)            #g  on obtien un sample
p_hat_VC <- evol_estim(y_VC)
IC_VC <- evol_IC(y_VC, p_hat_VC)
    

    #Resultats: (on note une reduction de varianc comme prévu!:)
    p_hat_VC[sample_size_n]
    IC_VC[sample_size_n,]

    
    
    
    
    
## Méthode d'estimation par allocation proportionnelle (AP)
    
estim_AP <- function(x_2, L){
  value <- 0
  variance <- 0
  n <- length(x_2)
  n_l <- c(n%/%L, diff((n * (1:L))%/%L))
  tmp <- 1  #va permetre de slicer notre x_2 en suivant le comportement de n.list
  tmp2 <- n_l[1]
  
  for (i in 1:L){  #on fait la loi condi
    
    
    u <- runif(n_l[i])
    x_2sliced <- x_2[tmp:tmp2]
    
    y <- qnorm((i - 1 + u)/L, mean = 0, sd = 2)   # retourne un vector de taille n1 = n/L
    y <- (exp(y) + exp(x_2sliced)) >= 5
    value <- value + sum(y)
    
    y <- as.integer(y)  #réglage technique
    
    if(length(y) > 1){
    variance <- variance + var(y)
    }
    
    
    tmp <- tmp + n_l[i]   #update pour le prochain tour, tmp commence à 1
    tmp2 <- tmp2 + n_l[i+1]
  }
  return(data.frame(value = value / n, variance = variance/(L*n)))
  
}
    
evolution_of_AP_estim <- function(L, x2 = f1_sample[,2]){

n <- length(f1_sample[,2])
if (n < L){
  stop("on veut que la taille du sample des X2 soit superieur ou egal a L")
}
level = 0.95
evolution <- c()
variance <- c()    #vont devenir des vect

  for (i in L:n) {
    evolution[i] <- estim_AP(f1_sample[,2][1:i], L)$value
    variance[i] <- estim_AP(f1_sample[,2][1:i], L)$variance
  }

bound.IC <- qnorm(0.5 * (level + 1)) * sqrt(variance/(1:n))  


return(data.frame(expect = evolution, var = variance, binf = evolution - bound.IC, bsup = evolution + bound.IC))

}           #on stock la méthode dans des dataframes

    
  #Lancer evolution_of_AP_estim, cela produira une dataframe,  par defaut le vecteur fixe des X2 générés est utilisé !
  

  results <- evolution_of_AP_estim(L=5)
  print(results[sample_size_n,])

    
    
  #Plot de l'évolution en fonction de n
  plot(1:sample_size_n, results$expect, ylim = c(0,1), type = "l", lwd = 1, col = 'skyblue', main = "Evolution de l'estimation AP de p")
  lines(2:sample_size_n, results$binf[2:sample_size_n], col = "firebrick", lwd = 1)  #plot des bornes de l'IC !
  lines(2:sample_size_n, results$bsup[2:sample_size_n], col = "firebrick", lwd = 1)  
  legend("topright", c("IC à 'level' %","estimation de p"), col = c("firebrick","skyblue"), lwd = c(2,1), pt.cex = 1, cex = 0.8, bty ="n")
  
    
    
    
    
    
    
    
    
    
### Second cas


## PARAMETRAGE TAILLE DU SAMPLE X = (Y1,Y2) ~ f_2

sample_size_n2 <- 5000
f2_sample <- method_reject_opti_f2(sample_size_n2)  

## PARAMETRAGE TAILLE DU SAMPLE X = (Y1,Y2) ~ f_2

    
# Méthode de Monte-Carlo classique (MC)
    
    y_MC_2 <- cos(f2_sample[,1] * f2_sample[,2]) * sin(f2_sample[,1]) * exp(sin(f2_sample[,1] + f2_sample[,2]))
    p_hat_MC_2 <- evol_estim(y_MC_2)
    IC_MC_2 <- evol_IC(y_MC_2, p_hat_MC_2)  # par defaut alpha = 5%
    
    #Estimation Monte-Carlo classique finale de la proba p; variance et IC associés
    p_hat_MC_2[sample_size_n2]
    IC_MC_2[sample_size_n2,]
    
    #Plot de l'évolution en fonction de n
    plot(1:sample_size_n2, p_hat_MC_2, ylim = c(0,1), type = "l", lwd = 1, col = 'skyblue', main = "Evolution de l'estimation classique de I")
    lines(2:sample_size_n2, IC_MC_2$binf[2:sample_size_n2], col = "firebrick", lwd = 1)  #plot des bornes de l'IC !
    lines(2:sample_size_n2, IC_MC_2$bsup[2:sample_size_n2], col = "firebrick", lwd = 1)  
    legend("topright", c("IC à 'level' %","estimation de p"), col = c("firebrick","skyblue"), lwd = c(2,1), pt.cex = 1, cex = 0.8, bty ="n")
    


    
# Méthode des variables antithétiques (VA)
    
    y_VA_2 <- 0.5 * ((cos(f2_sample[,1]*f2_sample[,2]) * sin(f2_sample[,1]) * exp(sin(f2_sample[,1] + f2_sample[,2]))) + (cos((-1) * f2_sample[,1] * (-1) * f2_sample[,2]) * sin(-1 * f2_sample[,1]) * exp(sin(-1 * f2_sample[,1] + -1 * f2_sample[,2]))   )    )                 #en fait on créé artificiellement de la nouvelle data (par transformation A)
    p_hat_VA_2 <- evol_estim(y_VA_2)
    IC_VA_2 <- evol_IC(y_VA_2, p_hat_VA_2)
    
    #Estimation finale,  de la proba p; variance et IC associés
    p_hat_VA_2[sample_size_n2]
    IC_VA_2[sample_size_n2,]
    
    #Plot de l'évolution en fonction de n
    plot(1:sample_size_n2, p_hat_VA_2, ylim = c(0,1), type = "l", lwd = 1, col = 'skyblue', main = "Evolution de l'estimation VA de I")
    lines(2:sample_size_n2, IC_VA_2$binf[2:sample_size_n2], col = "firebrick", lwd = 1)  #plot des bornes de l'IC !
    lines(2:sample_size_n2, IC_VA_2$bsup[2:sample_size_n2], col = "firebrick", lwd = 1)  
    legend("topright", c("IC à 'level' %","estimation de p"), col = c("firebrick","skyblue"), lwd = c(2,1), pt.cex = 1, cex = 0.8, bty ="n")
    

    
    
    
    
    
    



    
#################### PARTIE 3: Recyclage dans l'algorithme du rejet ####################
    
    
    
    
    method_reject_opti_f1_recycling <- function(N, f3_sample = 0){ #algo renvoie n realisation de densité f    //la realisation est acceptee en moyenne au temps c * m1 = 1/(pnorm(1)-pnorm(-1))
      ans <- c()       #ce que l'on va retourner, vide pour l'instant  //
      ans_rejected <- c()
      c <- 1 / ((pnorm(1) - pnorm(-1)) * 4 * pi)
      
      m1 <- 4 * pi 
      
      if (f3_sample == 1){
        m1 <- 6*pi
      }
      
      m1 <- 3* m1  # on va doubler car on rejette des valeurs qui ne passent aps
      
      compteur <- N   # nombre de réalisations, tant qu'on n'a pas atteint n
      # on poursuit l'algo tant qu'on n'a aps atteint n
      
        # en moyenne il faut c * m1 * n tour de boucle pour avoir n réalisations   /// floor(compteur * c * m1 + 1)
        
        unif <- runif(N) #les uniformes
        
        y <- matrix(rnorm(N * 2, mean = 0, sd = 1), ncol = 2) #on génère N = compteur*m1 ! couples (X1,X2) de N(0.1)
        
        #y suivent la loi de g
        y[,1] <- y[,1] * 2         #on ajuste la variance des X1 pour obtenir une N(0,4), on en a N ici
        
        #on applique le test de rejet ou non
        
        if (f3_sample == 0){
          test_vect <- ((m1 * unif * dnorm(y[,1], mean = 0, sd = 2) * dnorm(y[,2])) <= (f_tilde1(y[,1], y[,2])))
        }
        if (f3_sample == 1){
          test_vect <- ((m1 * unif * dnorm(y[,1], mean = 0, sd = 2) * dnorm(y[,2])) <= (f_tilde3(y[,1], y[,2])))
        }
        
        
        z <- test_vect * y
        z_rejected <- (!test_vect) * y
        # Z est un booléen (0 ou 1) * la fonction f !! car le Temps T coincide
        #en fait on garde les y qui vérifient le test, et ils suivent une loi suivant f !
        #test de la forme  m1*g*U <= f
        
        ans <- rbind(ans, z[(z[,1] != 0 | z[,2] != 0),]) #en faisant ça je voulais éviter le cas (improbable) où l'on ôte une variable alors qu'elle n'a pas été rejetée , parceque X1 ou X2 vaut 0
        ans_rejected <- rbind(ans_rejected, z_rejected[(z_rejected[,1] != 0 | z_rejected[,2] != 0),])
        
      
      return(list(f_accepted = ans, g_rejected = ans_rejected)) 
    }
    
    
    
    #Modifier la fonction ci-dessous pour un choix personnalisé de la fonction h
    
    h_function <- function(x, y){
      return((exp(x) + exp(y)) >= 5)  
    }
    
    f_1 <- function(x, y){
      
      c <- 1 / ((pnorm(1) - pnorm(-1)) * 4 * pi)
      
      return(c * f_tilde1(x,y))
      
    }
    
    g_1 <- function(x, y){ # densite d'une bivariate independante var 4 et var 1
      return(dnorm(x, mean = 0, sd = 2) * dnorm(y))
    }
    
    delta2_function <- function(x, y, M, g_density_instru, f_target_function, h = h_function){  
      
      for (i in 3:5) {  ## pour des tests techniques! 
        test <- (((M-1) * f_target_function(x , y)) / ((M * g_density_instru(x, y)) - f_target_function(x ,y)))
        
      }
        return ( (((M-1) * f_target_function(x , y)) / (M * g_density_instru(x, y) - f_target_function(x ,y))) )
      
    }
    
    
    
    
    evolution_of_deltas <- function(delta_number, alpha = 0, f3_sample = 0){
      
      
      if (!(f3_sample == 1 | f3_sample == 0)){
        print("on tire sous f3 oui ou non mon ami ? TAPER 1 ou 0")
      }
      
      
      
      level = 0.95
      
      evolution_delta1 <- c()
      variance_delta1 <- c()
      evolution_delta2 <- c()
      variance_delta2 <- c()
      evolution_delta3 <- c()
      variance_delta3 <- c()
      
      for (i in 1:N) {
        
        f_and_g <- method_reject_opti_f1_recycling(N, f3_sample)
        
        
        if (delta_number == 1){
          for (i in 1:N) {
            y_delta1 <- h_function(f_and_g$f_accepted[,1], f_and_g$f_accepted[,2])
            evolution_delta1[i] <- mean(y_delta1)
            variance_delta1[i] <- var(y_delta1)
          }
          bound.IC_delta1 <- qnorm(0.5 * (level + 1)) * sqrt(variance_delta1/(1:N))
          
          return(data.frame(expect = evolution_delta1, var = variance_delta1, binf = evolution_delta1 - bound.IC_delta1, bsup = evolution_delta1 + bound.IC_delta1))
        }
        
        if (delta_number == 2){
          for (i in 1:N) {
            y_delta2 <- h_function(f_and_g$g_rejected[,1], f_and_g$g_rejected[,2]) * delta2_function(x = f_and_g$g_rejected[,1], y = f_and_g$g_rejected[,2], M = 2 * 4 * pi, g_density_instru = g_1, f_target_function = f_1)
            evolution_delta2[i] <- mean(y_delta2)
            variance_delta2[i] <- var(y_delta2)
          }
          bound.IC_delta2 <- qnorm(0.5 * (level + 1)) * sqrt(variance_delta2/(1:N))
          
          return(data.frame(expect = evolution_delta2, var = variance_delta2, binf = evolution_delta2 - bound.IC_delta2, bsup = evolution_delta2 + bound.IC_delta2))
        }
          if (delta_number == 3){
            for (i in 1:N) {
              y_delta1 <- h_function(f_and_g$f_accepted[,1], f_and_g$f_accepted[,2])
              y_delta2 <- h_function(f_and_g$g_rejected[,1], f_and_g$g_rejected[,2]) * delta2_function(x = f_and_g$g_rejected[,1], y = f_and_g$g_rejected[,2], M = 2 * 4 * pi, g_density_instru = g_1, f_target_function = f_1)
              
             # y_delta3 <- alpha * y_delta1 + (1 - alpha) * y_delta2
              
              evolution_delta3[i] <- alpha * mean(y_delta1) + (1 - alpha) * mean(y_delta2)
              variance_delta3[i] <- (alpha^2) * var(y_delta1) + ((1 - alpha)^2) * var(y_delta2)
            }
            bound.IC_delta3 <- qnorm(0.5 * (level + 1)) * sqrt(variance_delta3/(1:N))
            
            return(data.frame(expect = evolution_delta3, var = variance_delta3, binf = evolution_delta3 - bound.IC_delta3, bsup = evolution_delta3 + bound.IC_delta3))
            
          
        }
      }
    }
    
    
    
    
     ## DEBUT PROGRAMME
    
    
    
    
    #Choix de la valeur de N !!
    
    N <- 2500  #attention, choisir un nombre assez petit(1000-3000) car l'algo est long surtout à cause de delta2
    
    
    
    dataframe_delta1 <- evolution_of_deltas(1, f3_sample = 0)
    
    print("Resultat final delta1")
    dataframe_delta1[N,]   # afin de voir le resultat
    
    plot(1:N, dataframe_delta1$expect, ylim = c(0,1), type = "l", lwd = 1, col = 'firebrick', main = "Evolution des estimateurs selon N (# of simulations)")
    lines(2:N, dataframe_delta1$binf[2:N], col = "firebrick", lwd = 1)  
    lines(2:N, dataframe_delta1$bsup[2:N], col = "firebrick", lwd = 1)  

    
    
    
    dataframe_delta2 <- evolution_of_deltas(2, f3_sample = 0)
    
    print("Resultat final delta2")
    dataframe_delta2[N,]
    
    lines(1:N, dataframe_delta2$expect, ylim = c(0,1), type = "l", lwd = 1, col = 'blue')
    lines(2:N, dataframe_delta2$binf[2:N], col = "blue", lwd = 1)  
    lines(2:N, dataframe_delta2$bsup[2:N], col = "blue", lwd = 1)  
    
    
    
    
    dataframe_delta3 <- evolution_of_deltas(3, alpha = 0.55, f3_sample = 0)
    
    lines(1:N, dataframe_delta3$expect, ylim = c(0,1), type = "l", lwd = 1, col = 'chartreuse4')
    lines(2:N, dataframe_delta3$binf[2:N], col = "chartreuse4", lwd = 1)  
    lines(2:N, dataframe_delta3$bsup[2:N], col = "chartreuse4", lwd = 1)  
  
    
    print("Resultat final delta3")
    dataframe_delta3[N,]
    
    
    legend("topright", c("delta_1 IC and estim","delta_2", "delta_3"), col = c("firebrick","blue", "chartreuse4"), lwd = c(2,1), pt.cex = 1, cex = 0.8, bty ="n")
    
    
    
    
    
    
    
    
    
    
    ########################### A CONSERVER ERREUR D APPROCHE #################
    
    
    #Génération d'un sample contenant N densité f ou g, retourne une list()
    
    f_and_g <- method_reject_opti_f1_recycling(N)
    
    #On determine alors notres valeur de T !t
    
    y_delta1 <- h_function(f_and_g$f_accepted[,1], f_and_g$f_accepted[,2])
    
    y_delta2 <- h_function(f_and_g$g_rejected[,1], f_and_g$g_rejected[,2]) * delta2_function(x = f_and_g$g_rejected[,1], y = f_and_g$g_rejected[,2], M = 2 * 4 * pi, g_density_instru = g_1, f_target_function = f_1)
      
  
    
    delta_1 <- evol_estim(y_delta1)               #estimateur classique sur les valeurs acceptees
    
    delta_2 <- evol_estim(y_delta2)
    
    
    IC_delta_1 <- evol_IC(x = y_delta1, delta_1)
    
    IC_delta_2 <- evol_IC(x = y_delta2, delta_2)    #  vect des moyenne empuiriques, liste des variances empiriques
    
    
    ########################### A CONSERVER ERREUR D APPROCHE #################
    
    
    
    
    

    
    
    
#################### PARTIE 4: Algorithme de Metropolis-Hastings (MH) ####################
    
    
    
    metropolis_hastings <- function(n, f_tilda = f_tilde1){  #Pour simuler selon f_2 mettre f_2 = TRUE
      
      x <- matrix(data = 0.2, ncol =2)
      
      
      for(i in 1:(n-1)){
        ksi <- c()
        ksi[1] <- rnorm(1, mean = 0, sd = 2) 
        ksi[2] <- rnorm(1, mean = 0, sd = 1)     #on génère ksi qui suit une bivariate sd = 2/ sd = 1
        
        alpha_proba <- min(1, ((f_tilda(ksi[1], (ksi[2])) * dnorm(x = x[i,1], sd = 2) * dnorm(x = x[i,2])) / (f_tilda(x[i,1], x[i,2]) * dnorm(x = ksi[1], sd = 2) * dnorm(x = ksi[2])))) 
        
        u <- runif(1)  #pour faire un test aleatoire
        
        if(u <= alpha_proba){
          x <- rbind(x, matrix(ksi,ncol = 2))      #MAJ de x (x de type matrix on met ksi de type matrix sinon on a des choses un peu bizarres)
        }else{
          x <- rbind(x,x[i,])
        }
      }
      
      return(x)
    }
    
    
    h_function <- function(x, y){  # pour calculer p
      return((exp(x) + exp(y)) >= 5)  
    }
  
    
    integrande_function <- function(x, y){
      
      return(cos(x * y) * sin(x) * exp(sin(x + y)))
      
    }
    
    
    
    
    evol_estim <- function(x){
      return(cumsum(x)/(1:length(x)))  #renvoie un vecteur de taille length(x)
    }
    
    evol_IC <- function(x, esperance, level = 0.95){   #retourne une data frame, esperance a le role d'un vect d'esperance
      n <- length(x)                
      s2 <- (cumsum(x^2) - (1:n) * esperance^2)/(0:(n-1)) # attention, cet estimateur n'est def que pour n >= 2 !
      bound.IC <- qnorm(0.5 * (level + 1)) * sqrt(s2/(1:n))  # q_1-alpha/2 * sqrt(varchap/n)
      return(data.frame(var = s2, binf = esperance - bound.IC, bsup = esperance + bound.IC))
    }
    
    
    
    ## Début de l'experience pour f_1 !
    
    
    
    n <- 5000
    
    
    markov_chain <- metropolis_hastings(n)
    
    y_MH <- h_function(markov_chain[,1], markov_chain[,2])
    
    p_hat_MH <- evol_estim(y_MH)
    IC_MH <- evol_IC(y_MH, p_hat_MH)  # par defaut alpha = 5%
    
    #Estimation Monte-Carlo classique finale de la proba p; variance et IC associés
    p_hat_MH[n]
    IC_MH[n,]
    
    #Plot de l'évolution en fonction de n
    plot(1:n, p_hat_MH, ylim = c(0,1), type = "l", lwd = 1, col = 'skyblue', main = "Evolution de l'estimation de p par MCMC")
    lines(2:n, IC_MH$binf[2:n], col = "firebrick", lwd = 1)  #plot des bornes de l'IC !
    lines(2:n, IC_MH$bsup[2:n], col = "firebrick", lwd = 1)  
    legend("topright", c("IC à 'level' %","estimation de p"), col = c("firebrick","skyblue"), lwd = c(2,1), pt.cex = 1, cex = 0.8, bty ="n")
    
    
    
    
    
    ## Début de l'experience pour f_2 !
    
    
    
    n <- 5000
    
    
    markov_chain_2 <- metropolis_hastings(n, f_tilda = f_tilde2)
    
    y_MH_2 <- integrande_function(markov_chain_2[,1], markov_chain_2[,2])
    
    p_hat_MH_2 <- evol_estim(y_MH_2)
    IC_MH_2 <- evol_IC(y_MH_2, p_hat_MH_2)  # par defaut alpha = 5%
    
    #Estimation Monte-Carlo classique finale de la proba p; variance et IC associés
    p_hat_MH_2[n]
    IC_MH_2[n,]
    
    #Plot de l'évolution en fonction de n
    plot(1:n, p_hat_MH_2, ylim = c(0,1), type = "l", lwd = 1, col = 'skyblue', main = "Evolution de l'estimation de I par MCMC")
    lines(2:n, IC_MH_2$binf[2:n], col = "firebrick", lwd = 1)  #plot des bornes de l'IC !
    lines(2:n, IC_MH_2$bsup[2:n], col = "firebrick", lwd = 1)  
    legend("topright", c("IC à 'level' %","estimation de p"), col = c("firebrick","skyblue"), lwd = c(2,1), pt.cex = 1, cex = 0.8, bty ="n")
    
    
    
    
    