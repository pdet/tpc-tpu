library("ggplot2")
library("ggthemes")

textsize <- 20
theme <- theme_few(base_size = textsize, base_family= "serif") + 
theme(legend.position="none",  axis.text=element_text(size=textsize, colour = "black"),
        axis.title=element_text(size=textsize, colour = "black"))


	# pdf(out, height=5, width=6)
	# print(ggplot(df, aes(x=sys, y=time_sec, group=sys)) + geom_bar(stat="identity", position = position_dodge(), width=.35, fill = "#777777") + xlab("") + ylab("Median time (s)") + theme + scale_x_discrete(labels=c(la, lb)) + geom_errorbar(aes(ymin=loconf, ymax=hiconf), width=.1, size=1.2) + geom_text(aes(label=round(time_sec, 2)), vjust=-.5, hjust=-.5, family="serif", size=7) )

	# dev.off() 


df <- data.frame(sys=c("CPU", "GPU", "TPU'", "HyPer", "CPU", "GPU", "TPU'", "HyPer"), time_sec=c(28, 0.68, 0.2, 7.5, NA, 5.06, 2.7, 16), grp=c("SF1","SF1","SF1","SF1","SF10","SF10","SF10","SF10"), stringsAsFactors=F)

df$sys <- ordered(df$sys, levels=c("CPU", "GPU", "TPU'", "HyPer"))

pdf("micro-filter.pdf", height=5, width=6)
print(ggplot(df, aes(x=sys, y=time_sec, group=sys, fill=sys)) + geom_bar(stat="identity", position = position_dodge(), width=.35) + xlab("") + ylab("Time (ms)") + theme + geom_text(aes(label=round(time_sec, 2)), vjust=-.5, hjust=.4, family="serif", size=7) + scale_y_continuous(limits=c(0, 35)) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
	+ facet_grid(~grp) ) 

dev.off() 



df <- data.frame(sys=c("CPU", "GPU", "TPU", "HyPer", "CPU", "GPU", "TPU", "HyPer"), time_sec=c(1.20,0.05,0.06,6.3,3.7,0.42,0.5,30), grp=c("SF1","SF1","SF1","SF1","SF10","SF10","SF10","SF10"), stringsAsFactors=F)

df$sys <- ordered(df$sys, levels=c("CPU", "GPU", "TPU", "HyPer"))

pdf("micro-aggr.pdf", height=5, width=6)
print(ggplot(df, aes(x=sys, y=time_sec, group=sys, fill=sys)) + geom_bar(stat="identity", position = position_dodge(), width=.35) + xlab("") + ylab("Time (ms)") + theme + geom_text(aes(label=round(time_sec, 2)), vjust=-.5, hjust=.4, family="serif", size=7) + scale_y_continuous(limits=c(0, 32)) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
	+ facet_grid(~grp) ) 

dev.off() 





df <- data.frame(sys=c("CPU", "GPU", "TPU", "HyPer", "CPU", "GPU", "TPU", "HyPer"), time_sec=c(13, 1.5, 0.12, 8.2,NA,14.4, 1.1, 14), grp=c("SF1","SF1","SF1","SF1","SF10","SF10","SF10","SF10"), stringsAsFactors=F)

df$sys <- ordered(df$sys, levels=c("CPU", "GPU", "TPU", "HyPer"))

pdf("micro-group.pdf", height=5, width=6)
print(ggplot(df, aes(x=sys, y=time_sec, group=sys, fill=sys)) + geom_bar(stat="identity", position = position_dodge(), width=.35) + xlab("") + ylab("Time (ms)") + theme + geom_text(aes(label=round(time_sec, 2)), vjust=-.5, hjust=.4, family="serif", size=7) + scale_y_continuous(limits=c(0, 16)) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
	+ facet_grid(~grp) ) 

dev.off() 




df <- data.frame(sys=c("CPU", "GPU", "TPU", "HyPer", "CPU", "GPU", "TPU", "HyPer"), time_sec=c(13.2, 5.42, 50, 7, 143, 47.1, NA, 20), grp=c("SF1","SF1","SF1","SF1","SF10","SF10","SF10","SF10"), stringsAsFactors=F)

df$sys <- ordered(df$sys, levels=c("CPU", "GPU", "TPU", "HyPer"))

pdf("micro-limit.pdf", height=5, width=6)
print(ggplot(df, aes(x=sys, y=time_sec, group=sys, fill=sys)) + geom_bar(stat="identity", position = position_dodge(), width=.35) + xlab("") + ylab("Time (ms)") + theme + geom_text(aes(label=round(time_sec, 2)), vjust=-.5, hjust=.4, family="serif", size=7) + scale_y_continuous(limits=c(0, 150)) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
	+ facet_grid(~grp) ) 

dev.off() 



df <- data.frame(sys=c("CPU", "GPU", "TPU", "HyPer", "CPU", "GPU", "TPU", "HyPer"), time_sec=c(1.2,10.7, 0.06, 1.6, 4, 12.7, 0.08, 0.9), grp=c("SF1","SF1","SF1","SF1","SF10","SF10","SF10","SF10"), stringsAsFactors=F)

df$sys <- ordered(df$sys, levels=c("CPU", "GPU", "TPU", "HyPer"))

pdf("micro-join.pdf", height=5, width=6)
print(ggplot(df, aes(x=sys, y=time_sec, group=sys, fill=sys)) + geom_bar(stat="identity", position = position_dodge(), width=.35) + xlab("") + ylab("Time (ms)") + theme + geom_text(aes(label=round(time_sec, 2)), vjust=-.5, hjust=.4, family="serif", size=7) + scale_y_continuous(limits=c(0, 15)) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
	+ facet_grid(~grp) ) 

dev.off() 





df <- data.frame(sys=c("CPU", "GPU", "TPU", "HyPer", "CPU", "GPU", "TPU", "HyPer"), time_sec=c(55, 8.6, 3.75, 10, NA, 80, 35, 40), grp=c("SF1","SF1","SF1","SF1","SF10","SF10","SF10","SF10"), stringsAsFactors=F)

df$sys <- ordered(df$sys, levels=c("CPU", "GPU", "TPU", "HyPer"))

pdf("q1.pdf", height=5, width=6)
print(ggplot(df, aes(x=sys, y=time_sec, group=sys, fill=sys)) + geom_bar(stat="identity", position = position_dodge(), width=.35) + xlab("") + ylab("Time (ms)") + theme + geom_text(aes(label=round(time_sec, 2)), vjust=-.5, hjust=.4, family="serif", size=7) + scale_y_continuous(limits=c(0, 90)) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
	+ facet_grid(~grp) ) 

dev.off() 





df <- data.frame(sys=c("CPU", "GPU", "TPU", "HyPer", "CPU", "GPU", "TPU", "HyPer"), time_sec=c(7.5, 1.2, 0.54, 5, NA, 19.5, 2.3, 23), grp=c("SF1","SF1","SF1","SF1","SF10","SF10","SF10","SF10"), stringsAsFactors=F)

df$sys <- ordered(df$sys, levels=c("CPU", "GPU", "TPU", "HyPer"))

pdf("q6.pdf", height=5, width=6)
print(ggplot(df, aes(x=sys, y=time_sec, group=sys, fill=sys)) + geom_bar(stat="identity", position = position_dodge(), width=.35) + xlab("") + ylab("Time (ms)") + theme + geom_text(aes(label=round(time_sec, 2)), vjust=-.5, hjust=.4, family="serif", size=7) + scale_y_continuous(limits=c(0, 25)) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
	+ facet_grid(~grp) ) 

dev.off() 

