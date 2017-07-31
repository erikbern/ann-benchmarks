package dk.itu.epbprtv0.distance;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;

import dk.itu.epbprtv0.Tokeniser;
import dk.itu.epbprtv0.types.DoubleInfo;

public class CosineDistance implements DistanceMetric<List<Double>> {
  public static final CosineDistance INSTANCE = new CosineDistance();

  @Override
  public double getDistance(List<Double> a, List<Double> b) {
    assert a.size() == b.size();
    int count = a.size();

    double dotProduct = 0.0;
    double as = 0.0, bs = 0.0;
    for (int i = 0; i < count; i++) {
      double ai = a.get(i), bi = b.get(i);
      dotProduct += ai * bi;
      as += Math.pow(ai, 2);
      bs += Math.pow(bi, 2);
    }

    return 1.0 - (dotProduct / (Math.sqrt(as) * Math.sqrt(bs)));
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
    List<String> tokens;
    while ((tokens = getTokens()) != null && !tokens.isEmpty()) {
      List<Double> p = DoubleInfo.INSTANCE.parse(tokens.get(0));
      List<Double> q = DoubleInfo.INSTANCE.parse(tokens.get(1));
      System.out.println(INSTANCE.getDistance(q, p));
    }
  }
}
