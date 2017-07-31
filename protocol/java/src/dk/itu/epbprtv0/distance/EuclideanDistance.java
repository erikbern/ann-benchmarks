package dk.itu.epbprtv0.distance;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;
import dk.itu.epbprtv0.Tokeniser;

public class EuclideanDistance implements DistanceMetric<List<Double>> {
  public static final EuclideanDistance INSTANCE = new EuclideanDistance();

  @Override
  public double getDistance(List<Double> q, List<Double> p) {
    assert q.size() == p.size();
    int count = q.size();

    double t = 0.0;
    for (int i = 0; i < count; i++)
      t += Math.pow(q.get(i) - p.get(i), 2);
    return Math.sqrt(t);
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
}
