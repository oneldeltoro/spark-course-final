package cu.uci.equipo1.app;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Optional;

import static org.apache.spark.sql.functions.col;

/**
 * <p> "Escriba su texto aquí"</p>
 * Author: Onel Del Toro Rodríguez <a href="mailto>:onel.deltoro@datys.cu">onel.deltoro@datys.cu</a>
 */
public class MultilayerPerceptronClassifier extends AlgorithmsBase{


    public static void main(String[] args) {
        SparkSession spark = getSparkSession();
        Dataset<Row> clean = getRowDatasetClean(spark, Optional.empty());

        //Dividimos los datos en dos partes 70 % para entrenar y 30 % para pruebas
        Dataset<Row>[] split = getDatasets(clean, Optional.of(new double[]{0.7, 0.3}), Optional.of(12345L));

        /**
         * Multilayer Perceptron
         */

        //Definimos la arquitectura con 6 neuronas en la capa de entrada (6 atributos)
        //4 y 3 como neuronas de las capa ocultas y 2 en la salida ya que son dos clasificaciones (efectuar cesarea o no)


        int[] layers = new int[]{6,5, 4, 3};

        org.apache.spark.ml.classification.MultilayerPerceptronClassifier redNeuronal = new org.apache.spark.ml.classification.MultilayerPerceptronClassifier()
                .setLayers(layers)
                .setBlockSize(128)
                .setSeed(1234L)
                .setMaxIter(100);
        redNeuronal.setFeaturesCol("featuresNormalized");
        redNeuronal.setLabelCol("label");

        //Discretizar la salida
        StringIndexer classIndexer = new StringIndexer().setInputCol("class").setOutputCol("label");

        VectorAssembler assembler = getVectorAssembler();

        Normalizer normalizer = new Normalizer()
                .setInputCol("features")
                .setOutputCol("featuresNormalized")
                .setP(1.0);

        Pipeline pipelineMLP = new Pipeline().setStages(
                new PipelineStage[]{
                        classIndexer,
                        assembler,
                        normalizer,
                        redNeuronal});

        //Configuramos el grid para buscar hiper-parámetros, en este caso de ejemplo máximo número de iteraciones
        ParamGridBuilder paramGridMLP = new ParamGridBuilder();
        paramGridMLP.addGrid(redNeuronal.stepSize(), new double[]{0.01, 0.001,0.0015});

        //Buscamos hiper-parámetros y ejecutamos el pipeline

        TrainValidationSplit trainValidationSplitMLP = getTrainValidationSplit(pipelineMLP, paramGridMLP);

        TrainValidationSplitModel modelMLP = trainValidationSplitMLP.fit(split[0]);
        Dataset<Row> resultMLP = modelMLP.transform(split[1]);

        resultMLP.show();
        //Analizar métricas de rendimiento Accuracy y Confusion matrix
        printResult(resultMLP);

    }
}
