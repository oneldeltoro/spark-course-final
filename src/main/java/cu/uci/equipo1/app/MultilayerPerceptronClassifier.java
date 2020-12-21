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

import static org.apache.spark.sql.functions.col;

/**
 * <p> "Escriba su texto aquí"</p>
 * Author: Onel Del Toro Rodríguez <a href="mailto>:onel.deltoro@datys.cu">onel.deltoro@datys.cu</a>
 */
public class MultilayerPerceptronClassifier {


    public static void main(String[] args) {

        /**
         *  Creando contexto y session de Apache Spark
         */
        SparkConf conf = new SparkConf().setAppName("Base de Columna Vertebral MultilayerPerceptron").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");

        //Para trabajar con Dataframes o Dataset(bd distribuidas)
        SparkSession spark = SparkSession.builder().sparkContext(sc.sc()).getOrCreate();

        /*
        Cree un Dataframe a partir del fichero e imprima en pantalla su esquema y las 10 primeras filas.
         */

        Dataset<Row> column_3c = spark.read().option("header", true).option("inferSchema", "true").csv("C:\\Users\\onel.deltoro\\IdeaProjects\\TareaFinal\\spark-course-final\\src\\main\\resources\\csv_result-column_3C_weka.csv");
        column_3c.printSchema();
        column_3c.show(10);

        /*
         Elimine las filas que tengan columnas con valores ausentes.
         */
        System.out.println("Limpiando los datos");
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

        /*Aplique las transformaciones necesarias sobre los datos
        que contengan valores nominales, mediante técnicas de extracción de características. */

        /**
         * Para este problema se necesita identificar los valores nominales
         * En mi dataSet las valiebles Nominales son:
         * @attribute pelvic_incidence
         * @attribute pelvic_tilt
         * @attribute lumbar_lordosis_angle
         * @attribute sacral_slope
         * @attribute pelvic_radius
         * @attribute degree_spondylolisthesis
         *
         * Estas son las necesarias para su conversion a variables Discretas. Las variables discretas son necesarias para
         * los datos de entrada.
         *
         */


        //Dividimos los datos en dos partes 70 % para entrenar y 30 % para pruebas
        Dataset<Row>[] split = clean.randomSplit(new double[]{0.7, 0.3}, 12345);
     /*   System.out.println("schema\n\n" + split[0].schema());
        System.out.println("schema\n\n" + split[0].schema().json());*/

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

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle"
                        , "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"})
                .setOutputCol("features");

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

        TrainValidationSplit trainValidationSplitMLP = new TrainValidationSplit()
                .setEstimator(pipelineMLP)
                .setEstimatorParamMaps(paramGridMLP.build())
                //Para el evaluador podemos elegir: BinaryClassificationEvaluator, ClusteringEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator
                .setEvaluator(new MulticlassClassificationEvaluator());

        TrainValidationSplitModel modelMLP = trainValidationSplitMLP.fit(split[0]);
        Dataset<Row> resultMLP = modelMLP.transform(split[1]);

        resultMLP.show();
        //Analizar métricas de rendimiento Accuracy y Confusion matrix
        MulticlassMetrics metrics3 = new MulticlassMetrics(resultMLP.select("prediction", "label"));

        System.out.println("Test set accuracy = " + metrics3.weightedFMeasure());
        System.out.println("Confusion matrix = \n" + metrics3.confusionMatrix());


    }
}
