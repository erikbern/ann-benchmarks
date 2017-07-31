package dk.itu.epbprtv0;

import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.List;

import dk.itu.epbprtv0.algorithm.LinearFactory;
import dk.itu.epbprtv0.algorithm.PreparedQueryAlgorithm;
import dk.itu.epbprtv0.algorithm.QueryAlgorithm;
import dk.itu.epbprtv0.algorithm.QueryAlgorithmFactory;

public abstract class Runner {
  private Runner() {
  }
  
  private static class Holder1 {
    public static final BufferedReader IN =
        new BufferedReader(new InputStreamReader(System.in));
  }
  protected static List<String> getTokens() {
    try {
      return Tokeniser.tokenise(Holder1.IN.readLine());
    } catch (IOException e) {
      return null;
    }
  }

  public static void main(String[] args) {
    QueryAlgorithmFactory fac = new LinearFactory();

    boolean preparedQueryMode = false;
    /* Configuration mode */
    List<String> tokens = getTokens();
    while (tokens.size() > 0) {
      if (tokens.size() == 2) {
        if (fac.configure(tokens.get(0), tokens.get(1))) {
          System.out.println("epbprtv0 ok");
        } else System.out.println("epbprtv0 fail");
      } else if (tokens.size() == 3 && tokens.get(0).equals("frontend")) {
        if (tokens.get(1).equals("prepared-queries")) {
          preparedQueryMode = (tokens.get(2).equals("1"));
          System.err.println("PCM is " + preparedQueryMode);
          System.out.println("epbprtv0 ok");
        } else System.out.println("epbprtv0 fail");
      } else System.out.println("epbprtv0 fail");
      tokens = getTokens();
    }

    QueryAlgorithm algo = fac.endConfigure();
    if (algo != null) {
      System.out.println("epbprtv0 ok");
    } else {
      System.out.println("epbprtv0 fail");
      return;
    }

    /* Training mode */
    tokens = getTokens();
    int success = 0, fail = 0;
    while (tokens.size() > 0) {
      if (tokens.size() == 1) {
        if (algo.train(tokens.get(0))) {
          success++;
          System.out.println("epbprtv0 ok");
        } else {
          fail++;
          System.out.println("epbprtv0 fail");
        }
      }
      tokens = getTokens();
    }

    algo.endTrain();
    System.out.print("epbprtv0 ok " + success);
    if (fail != 0) {
      System.out.println(" fail " + fail);
    } else System.out.println();

    tokens = getTokens();
    if (!preparedQueryMode) {
      while (tokens.size() > 0) {
        if (tokens.size() == 2) {
          try {
            String item = tokens.get(0);
            int k = Integer.parseInt(tokens.get(1));
            List<Integer> results = algo.query(item, k);
            if (results.size() > 0) {
              System.out.println("epbprtv0 ok " + results.size());
              for (int i : results)
                System.out.println("epbprtv0 " + i);
            } else System.out.println("epbprtv0 fail");
          } catch (NumberFormatException nfe) {
            System.out.println("epbprtv0 fail");
          }
        } else System.out.println("epbprtv0 fail");
        tokens = getTokens();
      }
      System.out.println("epbprtv0 ok");
    } else {
      PreparedQueryAlgorithm qAlgo = null;
      if (algo instanceof PreparedQueryAlgorithm)
        qAlgo = (PreparedQueryAlgorithm)algo;
      String lastItem = null;
      int lastK = -1;
      while (tokens.size() > 0) {
        if (tokens.size() == 1 && tokens.get(0).equals("query") &&
            lastK != -1 && lastItem != null) {
          List<Integer> results =
              algo.query(qAlgo != null ? null : lastItem, lastK);
          if (results.size() > 0) {
            System.out.println("epbprtv0 ok " + results.size());
            for (int i : results)
              System.out.println("epbprtv0 " + i);
          } else System.out.println("epbprtv0 fail");
        } else if (tokens.size() == 2) {
          try {
            lastItem = tokens.get(0);
            lastK = Integer.parseInt(tokens.get(1));
            if (qAlgo != null) {
              qAlgo.prepareQuery(lastItem);
              System.out.println("epbprtv0 ok prepared true");
            } else System.out.println("epbprtv0 ok prepared false");
          } catch (NumberFormatException nfe) {
            System.out.println("epbprtv0 fail");
          }
        } else System.out.println("epbprtv0 fail");
        tokens = getTokens();
      }
      System.out.println("epbprtv0 ok");
    }
  }
}