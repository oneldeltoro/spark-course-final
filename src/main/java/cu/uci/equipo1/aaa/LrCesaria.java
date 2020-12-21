/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cu.uci.equipo1.aaa;

import lombok.Getter;
import lombok.Setter;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.ml.tuning.TrainValidationSplitParams;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;

/**
 * @author liban
 */
public class LrCesaria implements IAaa {

    protected org.apache.spark.ml.classification.LogisticRegression lr
            = new org.apache.spark.ml.classification.LogisticRegression();

    @Getter
    @Setter
    public Pipeline pipeline;

    protected TrainValidationSplitModel model;

    protected Dataset<Row>[] split;

    @Getter
    @Setter
    protected double accuracy;

    public LrCesaria(String pathCsvFile) {

        SparkConf conf = new SparkConf().setAppName("Algoritmo de regresion logica").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");

        //Para trabajar con Dataframes o Dataset(bd distribuidas)
        SparkSession spark = SparkSession.builder().sparkContext(sc.sc()).getOrCreate();

        //crear dataset a partir de fichero csv
        Dataset<Row> df = spark.read().option("header", true).option("inferSchema", "true").csv(pathCsvFile);

        //para ver esquema y ver los primeros 10 registros
        df.printSchema();

        this.extractFeacture(df);
        this.practicing(df);
        Dataset<Row> rs = this.evaluate();

        rs.show();
    }


    public void LrCessaria() {

    }

    @Override
    public Pipeline extractFeacture(Dataset df) {

        //Preparamos las siguientes transformaciones, para datos nominales
        StringIndexer delivery_numberIndexer = new StringIndexer()
                .setInputCol("Delivery_number").setOutputCol("Delivery_numberIndex");
        StringIndexer delivery_timeIndexer = new StringIndexer()
                .setInputCol("Delivery_time").setOutputCol("Delivery_timeIndex");
        StringIndexer blood_PressureIndexer = new StringIndexer()
                .setInputCol("Blood_Pressure").setOutputCol("Blood_PressureIndex");
        StringIndexer heart_ProblemIndexer = new StringIndexer()
                .setInputCol("Heart_Problem").setOutputCol("Heart_ProblemIndex");

        //delivery_numberIndexer.fit(logredata).transform(logredata).show();
        OneHotEncoderEstimator encoder = new OneHotEncoderEstimator()
                .setInputCols(new String[]{"Delivery_numberIndex",
                        "Delivery_timeIndex",
                        "Blood_PressureIndex", "Heart_ProblemIndex"})
                .setOutputCols(new String[]{"Delivery_numberVec",
                        "Delivery_timeVec",
                        "Blood_PressureVec", "Heart_ProblemVec"});

//Creamos nuestro vector assembler con las columnas deseadas y la clase predictora
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"Age", "Delivery_number",
                        "Delivery_time", "Blood_Pressure", "Heart_Problem"})
                .setOutputCol("features");

        pipeline = new Pipeline().setStages(new PipelineStage[]{
                delivery_numberIndexer, delivery_timeIndexer, blood_PressureIndexer,
                heart_ProblemIndexer, encoder, assembler, lr});

        return pipeline;
    }

    @Override
    public TrainValidationSplitParams practicing(Dataset df) {
        Dataset<Row> logregdataall = df.select(col("Age"), col("Delivery_number"),
                col("Delivery_time"), col("Blood_Pressure"), col("Heart_Problem"),
                col("Cesarian").as("label"));
        logregdataall.show(10);

        //Eliminar valores ausentes
        Dataset<Row> logredata = logregdataall.na().drop();
        //Dividimos los datos en dos partes 70 % para entrenar y 30 % para pruebas
        split = logredata.randomSplit(new double[]{0.7, 0.3}, 12345);
        split[0].show(10);

        //Búsqueda de hiperparametros
        ParamGridBuilder paramGrid = new ParamGridBuilder();
        paramGrid.addGrid(lr.regParam(), new double[]{0.1, 0.01, 0.001, 0.0001});

        //Buscamos hiper-parámetros, en este caso buscamos el parámetro regularizador.
        TrainValidationSplit trainValidationSplitLR = new TrainValidationSplit()
                .setEstimator(pipeline)
                .setEvaluator(new RegressionEvaluator())
                .setEstimatorParamMaps(paramGrid.build())
                .setTrainRatio(0.8);

        //Ejecutamos el entrenamiento
        model = trainValidationSplitLR.fit(split[0]);

        return model;
    }

    @Override
    public Dataset<Row> evaluate() {
        //Ejecutamos las pruebas y lo guardamos en un dataset
        Dataset<Row> testResult = model.transform(split[1]);

//Evaluamos metricas de rendimiento a partir de las pruebas
        MulticlassClassificationEvaluator evaluator
                = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(testResult);
        System.out.println("Test Error = " + (1.0 - accuracy));
        return testResult;
    }

    public static void main(String[] arg) {
        String pathCsvFile = "src/main/resources/caesarian.csv";
        LrCesaria cesaria = new LrCesaria(pathCsvFile);
    }

}
