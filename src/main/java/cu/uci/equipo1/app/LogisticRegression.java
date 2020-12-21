package cu.uci.equipo1.app;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Optional;

/**
 * <p> "Escriba su texto aquí"</p>
 * Author: Onel Del Toro Rodríguez <a href="mailto>:onel.deltoro@datys.cu">onel.deltoro@datys.cu</a>
 */
public class LogisticRegression extends AlgorithmsBase {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession();
        Dataset<Row> clean = getRowDatasetClean(spark, Optional.empty());

        //Dividimos los datos en dos partes 70 % para entrenar y 30 % para pruebas
        Dataset<Row>[] split = getDatasets(clean, Optional.of(new double[]{0.7, 0.3}), Optional.of(12345L));

        VectorAssembler assembler = getVectorAssembler();

        StringIndexer classIndexer = new StringIndexer().setInputCol("class").setOutputCol("label");

        OneHotEncoderEstimator encoder = new OneHotEncoderEstimator()
                .setInputCols(new String[]{"label"})
                .setOutputCols(new String[]{"labelVec"});


        /**
         2-Seleccione al menos tres algoritmos de aprendizaje automático de acuerdo al problema identificado en el dataset y realice las siguientes acciones:
         * Para el Dataset del ejercicio determino que es un Problema de Clasificacion Multiclase
         * para este tipo de Problemas Spark propone o tiene implementado varios algoritmos, pero Yo escojo
         * 1-LogisticRegression
         * 2-MultilayerPerceptronClassifier
         * 3-RandomForestClassifier
         */


        //1-Creamos nuestro modelo de ML LogisticRegression

        org.apache.spark.ml.classification.LogisticRegression lr = new org.apache.spark.ml.classification.LogisticRegression();
        Pipeline pipeline = new Pipeline().setStages(
                new PipelineStage[]{
                        classIndexer,
                        encoder, assembler,
                        lr});

        //Búsqueda de hiperparametros
        ParamGridBuilder paramGrid = new ParamGridBuilder();
        paramGrid.addGrid(lr.regParam(), new double[]{0.1, 0.01, 0.001, 0.0001});

        //Buscamos hiper-parámetros, en este caso buscamos el parámetro regularizador.
        TrainValidationSplit trainValidationSplitLR = getTrainValidationSplit(pipeline, paramGrid);
        trainValidationSplitLR.setTrainRatio(0.8);

        //Ejecutamos el entrenamiento
        TrainValidationSplitModel model = trainValidationSplitLR.fit(split[0]);

        //Ejecutamos las pruebas y lo guardamos en un dataset
        Dataset<Row> testResult = model.transform(split[1]);
        testResult.show();
        printResult(testResult);
    }
}
