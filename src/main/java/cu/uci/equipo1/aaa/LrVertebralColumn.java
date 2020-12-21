/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cu.uci.equipo1.aaa;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitParams;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;

/**
 * @author liban
 */
public class LrVertebralColumn extends LrCesaria {

    public LrVertebralColumn(String pathCsvFile) {
        super(pathCsvFile);
    }

    @Override
    public Pipeline extractFeacture(Dataset df) {

//Creamos nuestro vector assembler con las columnas deseadas y la clase predictora
//Discretizar la salida
        StringIndexer classIndexer = new StringIndexer().setInputCol("class")
                .setOutputCol("label");

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"pelvic_incidence", "pelvic_tilt",
                        "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius",
                        "degree_spondylolisthesis",})
                .setOutputCol("features");

        Normalizer normalizer = new Normalizer()
                .setInputCol("features")
                .setOutputCol("featuresNormalized")
                .setP(1.0);

        pipeline = new Pipeline().setStages(new PipelineStage[]{classIndexer,
                assembler, lr});

        return pipeline;
    }

    @Override
    public TrainValidationSplitParams practicing(Dataset df) {
        //lr.setFeaturesCol("featuresNormalized");
        Dataset<Row> logregdataall = df.select(col("pelvic_incidence"),
                col("pelvic_tilt"), col("lumbar_lordosis_angle"),
                col("sacral_slope"), col("pelvic_radius"), col("degree_spondylolisthesis"),
                col("class"));
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

        setAccuracy(evaluator.evaluate(testResult));
        System.out.println("Test Error = " + (1.0 - getAccuracy()));
        return testResult;
    }

    public static void main(String[] arg) {
        String pathCsvFile = "src/main/resources/caesarian.csv";
        LrVertebralColumn logisticRegression
                = new LrVertebralColumn(pathCsvFile);

    }

}
