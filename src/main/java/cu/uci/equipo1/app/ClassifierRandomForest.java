package cu.uci.equipo1.app;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
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
public class ClassifierRandomForest {
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
         2-Seleccione al menos tres algoritmos de aprendizaje automático de acuerdo al problema identificado en el dataset y realice las siguientes acciones:
         * Para el Dataset del ejercicio determino que es un Problema de Clasificacion Multiclase
         * para este tipo de Problemas Spark propone o tiene implementado varios algoritmos, pero Yo escojo
         * 1-LogisticRegression
         * 2-MulticlassClassificationEvaluator
         * 3-LinearSVCModel
         */
        //Dividimos los datos en dos partes 70 % para entrenar y 30 % para pruebas
        Dataset<Row>[] split = clean.randomSplit(new double[]{0.7, 0.3}, 12345);
     /*   System.out.println("schema\n\n" + split[0].schema());
        System.out.println("schema\n\n" + split[0].schema().json());*/

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle"
                        , "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"})
                .setOutputCol("features");


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
        paramGrid.addGrid(rf.maxDepth(), new int[]{2,4,6,8,10});

        //Buscamos hiper-parámetros, en este caso buscamos el parámetro regularizador.
        TrainValidationSplit trainValidationSplitRF = new TrainValidationSplit()
                .setEstimator(pipeline)
                .setEvaluator(new MulticlassClassificationEvaluator())
                .setEstimatorParamMaps(paramGrid.build());

        //Ejecutamos el entrenamiento
        TrainValidationSplitModel model = trainValidationSplitRF.fit(split[0]);

        //Ejecutamos las pruebas y lo guardamos en un dataset
        Dataset<Row> testResult = model.transform(split[1]);
        testResult.show();

        //Analizar métricas de rendimiento Accuracy y Confusion matrix
        MulticlassMetrics metrics3 = new MulticlassMetrics(testResult.select("prediction", "label"));
        double accuracy = metrics3.weightedFMeasure();
        System.out.println("Test set accuracy = " +accuracy);
        System.out.println("Confusion matrix = \n" + metrics3.confusionMatrix());
        System.out.println("Test Error = " + (1.0 - accuracy));

    }
}
