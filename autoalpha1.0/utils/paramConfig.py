class paramConfig():
    def __init__(self,T):
        self.T= T#时间周期
        #回溯窗口
        self.lookbackT = 60 if (T=="H"or T=="D") else 60 * 2 if T=="30T" else 0
        #高度区域阈值
        self.hight_region = 0.6 if T== "D"else 0.3/4 if (T=="H" or T=="30T") else 0.3/8 if T=="15T" else 0
        #价格下降阈值
        self.dropDelta = 1 if T =="D" else 0.5 if(T=="H" or T=="30T") else 0
        #RSI
        self.rsiPeriodList = [2, 14]
        #rsi阈值
        self.rsi_threshold = [80]
        #移动平均线周期设置
        self.MAset = [40]
        #上下穿MA周期
        self.updownMAset = self.MAset[:2]