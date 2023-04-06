

#' @title Wavelet Based LSTM Model
#'
#' @param ts Time Series Data
#' @param MLag Maximum Lags
#' @param split_ratio Training and Testing Split
#' @param wlevels Wavelet Levels
#' @param epochs Number of epochs
#' @param LSTM_unit Number of unit in LSTM layer
#' @import caret dplyr caretForecast tseries stats wavelets TSLSTM
#' @return
#' \itemize{
#'   \item Train_actual: Actual train series
#'   \item Test_actual: Actual test series
#'   \item Train_fitted: Fitted train series
#'   \item Test_predicted: Predicted test series
#'   }
#' @export
#'
#' @examples
#' \donttest{
#'y<-rnorm(100,mean=100,sd=50)
#'WTSLSTM<-WaveletLSTM(ts=y)
#'}
#' @references
#' Paul, R.K. and Garai, S. (2021). Performance comparison of wavelets-based machine learning technique for forecasting agricultural commodity prices, Soft Computing, 25(20), 12857-12873

WaveletLSTM<-function(ts,MLag=12,split_ratio=0.8,wlevels=3,epochs=25,LSTM_unit=20){
  SigLags<-NULL
  SigLags<-function(Data,MLag){
    ts<-as.ts(na.omit(Data))
    adf1<-adf.test(na.omit(ts))
    if (adf1$p.value>0.05){
      ts<-ts
    } else {
      ts<-diff(ts)
    }
    adf2<-adf.test(ts)
    if (adf2$p.value>0.05){
      ts<-ts
    } else {
      ts<-diff(ts)
    }

    CorrRes<-NULL
    for (i in 1:MLag) {
      # i=1
      ts_y<-dplyr::lag(as.vector(ts), i)
      t<-cor.test(ts,ts_y)
      corr_res<-cbind(Corr=t$statistic,p_value=t$p.value)
      CorrRes<-rbind(CorrRes,corr_res)
    }
    rownames(CorrRes)<-seq(1:MLag)
    Sig_lags<-rownames(subset(CorrRes,CorrRes[,2]<=0.05))
    maxlag<-max(as.numeric(Sig_lags))
    return(list(Result=as.data.frame(CorrRes),SigLags=as.numeric(Sig_lags),MaxSigLag=maxlag))
  }
  ntest<-round(length(ts)*(1-split_ratio), digits = 0)
  Split1 <- caretForecast::split_ts(as.ts(ts), test_size = ntest)
  train_data1 <- Split1$train
  test_data1 <- Split1$test
  Wvlevels<-wlevels
  mraout <- wavelets::modwt(as.vector(ts), filter="haar", n.levels=Wvlevels)
  WaveletSeries <- cbind(do.call(cbind,mraout@W),mraout@V[[Wvlevels]])
  ts_fitted<-NULL
  ts_foreast<-NULL

  for (j in 1:ncol(WaveletSeries)) {
    w<-as.ts(WaveletSeries[,j])
    maxl<-SigLags(Data=w,MLag = MLag)$MaxSigLag
    model<-TSLSTM::ts.lstm(ts=w,xreg = NULL,tsLag=maxl,xregLag = 0,LSTMUnits=LSTM_unit, Epochs=epochs,SplitRatio =split_ratio)
    model_par<-rbind(model_par,model$Param)
    ts_fitted<-model$TrainFittedValue
    ts_foreast<-model$TestPredictedValue
  }

  trainf <- apply(ts_fitted,1,sum)
  testf <- apply(ts_foreast,1,sum)
  return(list(Train_actual=train_data1,Test_actual=test_data1,Train_fitted=trainf))
}
