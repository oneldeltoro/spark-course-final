package cu.uci.equipo1.app;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
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
@Slf4j
public class AlgorithmsBase {

    static String pathCsvFile = "src/main/resources/csv_result-column_3C_weka.csv";

    protected static SparkSession getSparkSession() {
        /**
         *  Creando contexto y session de Apache Spark
         */
        SparkConf conf = new SparkConf().setAppName("Base de Columna Vertebral").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");

        //Para trabajar con Dataframes o Dataset(bd distribuidas)
        return SparkSession.builder().sparkContext(sc.sc()).getOrCreate();
    }

    protected static Dataset<Row> getRowDatasetClean(SparkSession spark, Optional<String> pathCSV) {
    /*
    Cree un Dataframe a partir del fichero e imprima en pantalla su esquema y las 10 primeras filas.
     */
        String path = pathCSV.filter(p -> p != null).orElseGet(() -> pathCsvFile);
        Dataset<Row> column_3c = spark.read().option("header", true).option("inferSchema", "true").csv(path);
        column_3c.printSchema();
        column_3c.show(10);

        /*
         Elimine las filas que tengan columnas con valores ausentes.
         */
        log.info("Limpiando los datos");
        Dataset<Row> readCSV = column_3c.select(
                col("pelvic_incidence"),
                col("pelvic_tilt"),
                col("lumbar_lordosis_angle"),
                col("sacral_slope"),
                col("pelvic_radius"),
                col("degree_spondylolisthesis"),
                col("class"));

        Dataset<Row> clean = readCSV.na().drop();

        clean.createOrReplaceTempView("inciso1b");
        String query4 = "Select DISTINCT class from inciso1b ";
        System.out.println(" ejecutando consulta: " + query4);
        Dataset<Row> inciso4 = spark.sql(query4);
        inciso4.show();
        return clean;
    }

    protected static Dataset<Row>[] getDatasets(Dataset<Row> dataset, Optional<double[]> porcent, Optional<Long> seed) {

        return dataset.randomSplit(porcent.orElseGet(() -> new double[]{0.7, 0.3}), seed.orElseGet(() -> 12345L));
    }

    protected static VectorAssembler getVectorAssembler() {
        return new VectorAssembler()
                .setInputCols(new String[]{"pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle"
                        , "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"})
                .setOutputCol("features");
    }

    protected static void printResult(Dataset<Row> testResult) {
        //Analizar métricas de rendimiento Accuracy y Confusion matrix
        MulticlassMetrics metrics3 = new MulticlassMetrics(testResult.select("prediction", "label"));
        double accuracy = metrics3.weightedFMeasure();
        System.out.println("Test set accuracy = " + accuracy);
        System.out.println("Confusion matrix = \n" + metrics3.confusionMatrix());
        System.out.println("Test Error = " + (1.0 - accuracy));
    }

    protected static TrainValidationSplit getTrainValidationSplit(Pipeline pipeline, ParamGridBuilder paramGrid) {
        //Buscamos hiper-parámetros, en este caso buscamos el parámetro regularizador.
        return new TrainValidationSplit()
                .setEstimator(pipeline)
                //Para el evaluador podemos elegir: BinaryClassificationEvaluator, ClusteringEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator
                .setEvaluator(new MulticlassClassificationEvaluator())
                .setEstimatorParamMaps(paramGrid.build());
    }

}
