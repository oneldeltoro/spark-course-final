package cu.uci.equipo1.app;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.feature.IndexToString;
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
public class ClassifierRandomForest extends AlgorithmsBase {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession();
        Dataset<Row> clean = getRowDatasetClean(spark, Optional.empty());

        /*Aplique las transformaciones necesarias sobre los datos
        que contengan valores nominales, mediante técnicas de extracción de características. */

        //Dividimos los datos en dos partes 70 % para entrenar y 30 % para pruebas
        Dataset<Row>[] split = getDatasets(clean, Optional.of(new double[]{0.7, 0.3}), Optional.of(12345L));

        VectorAssembler assembler = getVectorAssembler();

        StringIndexer classIndexer = new StringIndexer().setInputCol("class").setOutputCol("label");

        // Entrena un modelo RandomForest.
        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features");

// Convert indexed labels back to original labels.
        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(classIndexer.fit(clean).labels());

// Chain indexers and forest in a Pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{classIndexer,
                        assembler,
                        rf,
                        labelConverter});

        //Búsqueda de hiperparametros
        ParamGridBuilder paramGrid = new ParamGridBuilder();
        paramGrid.addGrid(rf.subsamplingRate(), new double[]{0.1, 0.01, 0.001, 0.0001});
        paramGrid.addGrid(rf.maxDepth(), new int[]{2, 4, 6, 8, 10});
        TrainValidationSplit trainValidationSplitRF = getTrainValidationSplit(pipeline, paramGrid);

        //Ejecutamos el entrenamiento
        TrainValidationSplitModel model = trainValidationSplitRF.fit(split[0]);

        //Ejecutamos las pruebas y lo guardamos en un dataset
        Dataset<Row> testResult = model.transform(split[1]);
        testResult.show();
        printResult(testResult);

    }

}
