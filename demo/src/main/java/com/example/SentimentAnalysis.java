package com.example;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.IOException;
import java.util.Properties;

public class SentimentAnalysis {

    public static class SentimentMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        private StanfordCoreNLP pipeline;

        @Override
        protected void setup(Context context) {
            Properties props = new Properties();
            props.setProperty("annotators", "tokenize, ssplit, parse, sentiment");
            pipeline = new StanfordCoreNLP(props);
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String text = value.toString();
            Annotation annotation = pipeline.process(text);
            
            // Compute the average and convert to int, acknowledging that this truncates decimal values.
            int sentiment = (int) annotation.get(CoreAnnotations.SentencesAnnotation.class)
                .stream()
                .mapToInt(sent -> RNNCoreAnnotations.getPredictedClass(sent.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class)))
                .average()
                .orElse(-1); // Keep as -1, since we are working with ints here.

            context.write(new Text("Average Sentiment"), new IntWritable(sentiment));
        }
    }

    public static class SentimentReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            int count = 0;
            for (IntWritable val : values) {
                sum += val.get();
                count++;
            }
            context.write(key, new IntWritable(sum / count)); // Write out the average sentiment
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length < 2) {
            System.err.println("Usage: sentimentanalysis <in> <out>");
            System.exit(2);
        }
        Job job = Job.getInstance(conf, "sentiment analysis");
        job.setJarByClass(SentimentAnalysis.class);
        job.setMapperClass(SentimentMapper.class);
        job.setCombinerClass(SentimentReducer.class);
        job.setReducerClass(SentimentReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}

