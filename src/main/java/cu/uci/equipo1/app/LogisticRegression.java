package cu.uci.equipo1.app;

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
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;

/**
 * <p> "Escriba su texto aquí"</p>
 * Author: Onel Del Toro Rodríguez <a href="mailto>:onel.deltoro@datys.cu">onel.deltoro@datys.cu</a>
 */
public class LogisticRegression {
    public static void main(String[] args) {

        /**
         *  Creando contexto y session de Apache Spark
         */
        SparkConf conf = new SparkConf().setAppName("Base de Columna Vertebral").setMaster("local[*]");
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
        StringIndexer classIndexer = new StringIndexer().setInputCol("class").setOutputCol("label");

        OneHotEncoderEstimator encoder = new OneHotEncoderEstimator()
                .setInputCols(new String[]{"label"})
                .setOutputCols(new String[]{"labelVec"});
        //Creamos nuestro vector assembler con las columnas deseadas y la clase predictora
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle"
                        , "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"})
                .setOutputCol("features");

        //Dividimos los datos en dos partes 70 % para entrenar y 30 % para pruebas
        //Dividimos los datos en dos partes 70 % para entrenar y 30 % para pruebas
        Dataset<Row>[] split = clean.randomSplit(new double[]{0.7, 0.3}, 12345);
       // split[0].show(10);
        System.out.println("schema\n\n" + split[1].schema());
        System.out.println("schema\n\n" + split[1].schema().json());
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
        TrainValidationSplit trainValidationSplitLR = new TrainValidationSplit()
                .setEstimator(pipeline)
                .setEvaluator(new RegressionEvaluator())
                .setEstimatorParamMaps(paramGrid.build())
                .setTrainRatio(0.8);

        //Ejecutamos el entrenamiento
        TrainValidationSplitModel model = trainValidationSplitLR.fit(split[0]);

        //Ejecutamos las pruebas y lo guardamos en un dataset
        Dataset<Row> testResult = model.transform(split[1]);
        testResult.show();
        //Evaluamos metricas de rendimiento a partir de las pruebas
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(testResult);
        System.out.println("accuracy is: " + accuracy);
        System.out.println("Test Error = " + (1.0 - accuracy));
    }
}
