import re
import operator
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.types import *
from pyspark.sql.functions import udf,concat,explode, col, lit


conf = SparkConf().setAppName("Entity Resolution")
sc = SparkContext(conf=conf)
sqlCt = SQLContext(sc)

def convert(row,stops):
				words=re.split(r'\W+', row)
				words=map(lambda a:a.lower(),words)  
				completewords = []
				for i in (words):
					if i not in stops:
						if i!='':
							completewords.append(i)
				return completewords

class EntityResolution:
    def __init__(self, dataFile1, dataFile2, stopWordsFile):
        self.f = open(stopWordsFile, "r")
        self.stopWords = set(self.f.read().split("\n"))
        self.stopWordsBC = sc.broadcast(self.stopWords).value
        self.df1 = sqlCt.read.parquet(dataFile1).cache()
        self.df2 = sqlCt.read.parquet(dataFile2).cache()

    def preprocessDF(self, df, cols):
		df=df.withColumn('joincolumn', concat(col(cols[0]), lit(" "), col(cols[1])))
		stopwordlist=self.stopWordsBC
		UDFconvert = udf(lambda p :convert(p,stopwordlist),ArrayType(StringType()))
		df=df.withColumn("joinKey", UDFconvert(df.joincolumn))
		return df	
		
    def filtering(self, df1, df2):
     
        df1 = df1.withColumn("token", explode(df1.joinKey)).cache()
        df2 = df2.withColumn("token", explode(df2.joinKey)).cache()
        candDF=(df2.join(df1 ,df1.token==df2.token)
                           .select(df1.id.alias("id1"), df1.joinKey.alias("joinKey1"), 
                df2.id.alias("id2"), df2.joinKey.alias("joinKey2")).distinct())
        return candDF
		
    def verification(self, candDF, threshold):
        jaccard_similarity = (udf(lambda a,b: 1.0*len(set(a).intersection(set(b))) /len(set(a).union(set(b))), FloatType()))
        resultDF = candDF.withColumn("jaccard", jaccard_similarity(candDF.joinKey1, candDF.joinKey2))
        return resultDF.where(resultDF.jaccard >= threshold)
		
    def evaluate(self, result, groundTruth):
        precision=0.0
        recall=0.0
        fmeasure=0.0
        tmatrix = []
        for i in (result):
            if i in groundTruth:
             tmatrix.append(i)
        T=len(tmatrix)
        R=len(result)
        A=len(groundTruth)
        precision = 1.0*T/R
        recall =1.0*T/A
        fmeasure=2.0*precision*recall/(precision+recall)
        return precision, recall, fmeasure
		
			
    def jaccardJoin(self, cols1, cols2, threshold):
        newDF1 = self.preprocessDF(self.df1, cols1)
        newDF2 = self.preprocessDF(self.df2, cols2)
        print "Before filtering: %d pairs in total" %(self.df1.count()*self.df2.count()) 
        candDF = self.filtering(newDF1, newDF2)
        print "After Filtering: %d pairs left" %(candDF.count())
        
        resultDF=self.verification(candDF,threshold)
        print "After Verification: %d similar pairs" %(resultDF.count())
        
        return resultDF
       
		
    def __del__(self):
        self.f.close()


if __name__ == "__main__":
	er = EntityResolution("amazon-google/Amazon", "amazon-google/Google", "amazon-google/stopwords.txt")
	amazonCols = ["title", "manufacturer"]
	googleCols = ["name", "manufacturer"]
	resultDF = er.jaccardJoin(amazonCols, googleCols, 0.5)
	
	result = resultDF.map(lambda row: (row.id1, row.id2)).collect()
	groundTruth = sqlCt.read.parquet("amazon-google/Amazon_Google_perfectMapping") \
                          .map(lambda row: (row.idAmazon, row.idGoogle)).collect()
	
	print "(precision, recall, fmeasure) = ", er.evaluate(result, groundTruth)