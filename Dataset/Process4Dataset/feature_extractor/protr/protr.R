protr <- function (cache_file){
    library(protr)
    setwd(cache_file)
    selected <- readFASTA(paste0(cache_file, "/fasta_file.fasta"))
    # Common descriptors
    # ACC -- 20
    acc <- t(sapply(selected, extractAAC))
    # write.table(acc,
    #             file=paste0(cache_file, "/ACC.csv"),
    #             row.names = FALSE,
    #             col.names = FALSE)
    # DC -- 400
    dc <- t(sapply(selected, extractDC))
    # write.table(dc,
    #             file=paste0(cache_file, "/DC.csv"),
    #             row.names = FALSE,
    #             col.names = FALSE)
    # TC -- 8000 -D
    tc <- t(sapply(selected, extractTC))
    # write.table(tc,
    #             file=paste0(cache_file, "/TC.csv"),
    #             row.names = FALSE,
    #             col.names = FALSE)
    # MoreauBroto -- 240 -D
    moreau <- t(sapply(selected, extractMoreauBroto))
    # write.table(moreau,
    #             file=paste0(cache_file, "/MoreauBrato.csv"),
    #             row.names = FALSE,
    #             col.names = FALSE)
    # Moran -- 240 -D
    moran <- t(sapply(selected, extractMoran))
    # write.table(moran,
    #             file=paste0(cache_file, "/Moran.csv"),
    #             row.names = FALSE,
    #             col.names = FALSE)
    # Geary -- 240 -D
    geary <- t(sapply(selected, extractGeary))
    # write.table(geary,
    #             file=paste0(cache_file, "/Geary.csv"),
    #             row.names = FALSE,
    #             col.names = FALSE)
    # CTDC -- 21
    ctdc <- t(sapply(selected, extractCTDC))
    # write.table(ctdc,
    #             file=paste0(cache_file, "/CTDC.csv"),
    #             row.names = FALSE,
    #             col.names = FALSE)
    # CTDT -- 21
    ctdt <- t(sapply(selected, extractCTDT))
    # write.table(ctdt,
    #             file=paste0(cache_file, "/CTDT.csv"),
    #             row.names = FALSE,
    #             col.names = FALSE)
    # CTDD -- 105
    ctdd <- t(sapply(selected, extractCTDD))
    # write.table(ctdd,
    #             file=paste0(cache_file, "/CTDD.csv"),
    #             row.names = FALSE,
    #             col.names = FALSE)

    # CTtiad -- 343 -D
    ctriad <- t(sapply(selected, extractCTriad))
    # write.table(ctriad,
    #             file=paste0(cache_file, "/CTriad.csv"),
    #             row.names = FALSE,
    #             col.names = FALSE)
    # SOCN -- 60 -D
    socn <- t(sapply(selected, extractSOCN))
    # write.table(socn,
    #             file=paste0(cache_file, "/SOCN.csv"),
    #             row.names = FALSE,
    #             col.names = FALSE)
    # QSO -- 100
    qso <- t(sapply(selected, extractQSO))
    # write.table(qso,
    #             file=paste0(cache_file, "/QSO.csv"),
    #             row.names = FALSE,
    #             col.names = FALSE)
    # PAAC -- 50
    paac <- t(sapply(selected, extractPAAC))
    # write.table(paac,
    #             file=paste0(cache_file, "/PAAC.csv"),
    #             row.names = FALSE,
    #             col.names = FALSE)
    # APAAC -- 80
    apaac <- t(sapply(selected, extractAPAAC))
    # write.table(apaac,
    #             file=paste0(cache_file, "/APAAC.csv"),
    #             row.names = FALSE,
    #             col.names = FALSE)
    # Whole -- 797
    # whole <- c(acc, dc, tc, moreau, moran, geary, ctdc, ctdt, ctdd, ctriad, socn, qso, paac, apaac)
    # write.table(whole,
    #             file=paste0(cache_file, "/WholeFeatures.csv"),
    #             row.names = FALSE,
    #             col.names = FALSE)
    return(list(ACC=acc, DC=dc, TC=tc, MOREAU=moreau, MORAN=moran, GEARY=geary, CTDC=ctdc,
                CTDT=ctdt, CTDD=ctdd, CTRIAD=ctriad, SOCN=socn, QSO=qso, PAAC=paac, APAAC=apaac))
}
result <- protr("D:/WorkPath/PycharmProjects/MutTm-pred/Dataset/Process4Dataset/feature_extractor/protr")