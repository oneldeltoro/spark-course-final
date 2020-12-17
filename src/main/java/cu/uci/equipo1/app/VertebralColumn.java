package cu.uci.equipo1.app;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import static org.apache.spark.sql.functions.col;
/**
 * <p> "Escriba su texto aquí"</p>
 * Author: Onel Del Toro Rodríguez <a href="mailto>:onel.deltoro@datys.cu">onel.deltoro@datys.cu</a>
 */
public class VertebralColumn {
    public static void main(String[] args) {

        /**
         *  Creando contexto y session de Apache Spark
         */
        SparkConf conf = new SparkConf().setAppName("Base de datos cesarea").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        sc.setLogLevel("ERROR");

        //Para trabajar con Dataframes o Dataset(bd distribuidas)
        SparkSession spark = SparkSession.builder().sparkContext(sc.sc()).getOrCreate();

        /*
        Cree un Dataframe a partir del fichero e imprima en pantalla su esquema y las 10 primeras filas.
         */

        System.out.println("Cree un Dataframe a partir del fichero e imprima en pantalla su esquema y las 10 primeras filas");
        Dataset<Row> column_2c = spark.read().option("header", true).option("inferSchema", "true").csv("C:\\Users\\onel.deltoro\\IdeaProjects\\TareaFinal\\spark-course-final\\src\\main\\resources\\csv_result-column_2C_weka.csv");
        column_2c.printSchema();
        column_2c.show(10);

        Dataset<Row> column_3c = spark.read().option("header", true).option("inferSchema", "true").csv("C:\\Users\\onel.deltoro\\IdeaProjects\\TareaFinal\\spark-course-final\\src\\main\\resources\\csv_result-column_3C_weka.csv");
        column_3c.printSchema();
        column_3c.show(10);
    }
}
