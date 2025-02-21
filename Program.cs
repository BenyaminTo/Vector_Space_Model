using Accord.Math;
using System.Collections.Generic;
using System.Linq;
using System;

class Program
{
    static void Main(string[] args)
    {
        // نمونه اسناد و پرس و جو
        string[] documents = {
            "The financial crisis in Ghana was caused by poor economic policies.",
            "Ghana's banking sector faced a severe crisis in 2023.",
            "The United States is a major global economy.",
            "Mathematics is a fundamental subject in education.",
            "Ghana is a country in West Africa."
        };
        string query = "what caused the financial crisis in Ghana?";

        // پیش پردازش اسناد و پرس و جو
        string[] preprocessedDocs = documents.Select(doc => PreprocessText(doc)).ToArray();
        string preprocessedQuery = PreprocessText(query);

        // به روز کردن پرس و جو با اصطلاحات معنایی مرتبط 
        string updatedQuery = UpdateQuery(preprocessedQuery, preprocessedDocs);

        // محاسبه فرکانس های ترم و وزن های تناسب   
        double[,] tfP;
        double[] queryTf;
        ComputeTfP(preprocessedDocs, updatedQuery, out tfP, out queryTf);

        // محاسبه اندازه اسناد
        int[] documentSizes = preprocessedDocs.Select(doc => doc.Split(' ').Length).ToArray();

        // محاسبه شباهت کسینوس توسعه یافته 
        double[] cosineSim = ExtendedCosineSimilarity(tfP, queryTf, documentSizes);

        // رتبه بندی اسناد بر اساس شباهت کسینوس
        var rankedIndices = cosineSim.Select((value, index) => new { Value = value, Index = index })
                                     .OrderByDescending(x => x.Value)
                                     .Select(x => x.Index)
                                     .ToArray();
        //Print Resaults
        var rankedDocuments = rankedIndices.Select(i => documents[i]).ToArray();

        Console.WriteLine("Query:what caused the financial crisis in Ghana?");
        Console.WriteLine("Ranked Documents:");
        for (int i = 0; i < rankedDocuments.Length; i++)
        {
            Console.WriteLine($"{i + 1}.{rankedDocuments[i]}");
        }
    }//End of Main

    //Preprocessing Function
    static string PreprocessText(string text)
    {
        // نشانه گذاری و حذف علائم نگارشی
        var tokens = text.ToLower().Split(new[] { ' ', '.', ',', '!', '?', ';', ':', '"', '(', ')' }, StringSplitOptions.RemoveEmptyEntries);
        // Remove stopwords
        var stopWords = new HashSet<string> { "a", "an", "the", "and", "or", "but", "is", "in", "it", "to", "of", "for", "on", "with", "as", "by", "at" };
        tokens = tokens.Where(word => !stopWords.Contains(word)).ToArray();
        return string.Join(" ", tokens);
    }

    static string UpdateQuery(string query, string[] documents, double threshold = 0.65)
    {
        //تبدیل متن کوئری و اسناد به بردار تی اف پی 
        var vectorizer = new TfpVectorizer();
        var tfpMatrix = vectorizer.FitTransform(documents);
        var queryVec = vectorizer.Transform(new[] { query });
        // محاسبه شباهت کسینوس بین پرس و جو و اسناد
        var similarities = tfpMatrix.ToArray().Select(docVec => CosineSimilarity(queryVec.ToArray()[0], docVec)).ToArray();
        // محاسبه عباراتی که از نظر معنایی مشابه پرس و جو هستند
        var updatedQuery = new HashSet<string>(query.Split(' '));
        for (int i = 0; i < documents.Length; i++)
        {
            if (similarities[i] >= threshold)//اگر شباهت بیشتر از آستانه کوئری باشد
            {
                //کلمات آن سند به کوئری اضافه شود
                var docTerms = documents[i].Split(' ');
                updatedQuery.UnionWith(docTerms);
            }
        }
        return string.Join(" ", updatedQuery);
    }

    //محاسبه وزن های تی اف پی برای اسناد و کوئری به روزرسانی شده
    static void ComputeTfP(string[] documents, string updatedQuery, out double[,] tfP, out double[] queryTf)
    {
        //تبدیل متن کوئری و اسناد به بردار تی اف پی
        var vectorizer = new TfpVectorizer();
        var tfpMatrix = vectorizer.FitTransform(documents);
        var queryVec = vectorizer.Transform(new[] { updatedQuery });
        // محاسبه فرکانس های ترم 
        tfP = tfpMatrix;
        queryTf = queryVec.ToArray()[0];
        // محاسبه وزن تناسب
        double[] p = Divide(queryTf, SumRows(tfP));
        // tf-p محاسبه وزن
        for (int i = 0; i < tfP.GetLength(0); i++)
        {
            for (int j = 0; j < tfP.GetLength(1); j++)
            {
                tfP[i, j] *= p[i];
            }
        }
    }

    //محاسبه شباهت کسینوسی توسعه یافته بین کوئری و مجموعه ای از اسناد
    static double[] ExtendedCosineSimilarity(double[,] tfP, double[] queryTf, int[] documentSizes)
    {
        // p_qd محاسبه وزن تناسب 
        double[] pQd = Divide(queryTf, documentSizes.Select(x => (double)x).ToArray());
        // محاسبه شباهت کسینوس توسعه یافته
        double[] cosineSim = new double[tfP.GetLength(0)];
        for (int i = 0; i < tfP.GetLength(0); i++)
        {
            double dotProduct = DotProduct(tfP.GetRow(i), queryTf);
            double normTfP = Normalize(tfP.GetRow(i));
            double normQueryTf = Normalize(queryTf);
            cosineSim[i] = (dotProduct / (normTfP * normQueryTf)) * pQd[i];
        }
        return cosineSim;
    }

    //محاسبه شباهت کسینوسی بین دو بردار
    static double CosineSimilarity(double[] vecA, double[] vecB)
    {
        double dotProduct = DotProduct(vecA, vecB);
        double normA = Normalize(vecA);
        double normB = Normalize(vecB);
        return dotProduct / (normA * normB);
    }

    static double[] Divide(double[] a, double[] b)
    {
        return a.Zip(b, (x, y) => x / y).ToArray();
    }

    static double[] SumRows(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        double[] result = new double[rows];
        for (int i = 0; i < rows; i++)
        {
            double sum = 0;
            for (int j = 0; j < cols; j++)
            {
                sum += matrix[i, j];
            }
            result[i] = sum;
        }
        return result;
    }

    static double DotProduct(double[] a, double[] b)
    {
        return a.Zip(b, (x, y) => x * y).Sum();
    }

    static double Normalize(double[] vector)
    {
        double sum = vector.Sum(x => x * x);
        return Math.Sqrt(sum);
    }
}

public class TfpVectorizer
{
    private List<string> vocabulary;
    private double[] p;

    public double[,] FitTransform(string[] documents)
    {
        // Build vocabulary
        vocabulary = documents.SelectMany(doc => doc.Split(' ')).Distinct().ToList();

        // محاسبه فرکانس های ترم (TF)
        var tf = new double[documents.Length, vocabulary.Count];
        for (int i = 0; i < documents.Length; i++)
        {
            var terms = documents[i].Split(' ');
            for (int j = 0; j < vocabulary.Count; j++)
            {
                tf[i, j] = terms.Count(t => t == vocabulary[j]);
            }
        }

        // محاسبه p
        p = new double[vocabulary.Count];
        for (int j = 0; j < vocabulary.Count; j++)
        {
            int df = documents.Count(doc => doc.Split(' ').Contains(vocabulary[j]));
            p[j] = Math.Log((double)documents.Length / df);
        }

        // محاسبه TF-P
        var tfp = new double[documents.Length, vocabulary.Count];
        for (int i = 0; i < documents.Length; i++)
        {
            for (int j = 0; j < vocabulary.Count; j++)
            {
                tfp[i, j] = tf[i, j] * p[j];
            }
        }
        return tfp;
    }

    public double[,] Transform(string[] documents)
    {
        var tfp = new double[documents.Length, vocabulary.Count];
        for (int i = 0; i < documents.Length; i++)
        {
            var terms = documents[i].Split(' ');
            for (int j = 0; j < vocabulary.Count; j++)
            {
                int tf = terms.Count(t => t == vocabulary[j]);
                tfp[i, j] = tf * p[j];
            }
        }
        return tfp;
    }
}