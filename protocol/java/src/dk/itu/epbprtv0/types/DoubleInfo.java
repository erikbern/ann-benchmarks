package dk.itu.epbprtv0.types;

import dk.itu.epbprtv0.Tokeniser;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

public class DoubleInfo implements TypeInfo<List<Double>> {
  public static final DoubleInfo INSTANCE = new DoubleInfo();

  @Override
  public List<Double> parse(String item) {
    ArrayList<Double> results = new ArrayList<Double>();
    for (String s : Tokeniser.tokenise(item))
      results.add(Double.parseDouble(s));
    return results;
  }

  @Override
  public String unparse(List<Double> item) {
    StringBuilder sb = new StringBuilder();
    Iterator<Double> it = item.iterator();
    while (it.hasNext()) {
      sb.append(it.next().toString());
      if (it.hasNext())
        sb.append(' ');
    }
    return sb.toString();
  }

  public static void main(String[] args) {
    List<Double> h = DoubleInfo.INSTANCE.parse("   0.0120 1.0e-4 40059 723897 43.4378 342489237 4");
    System.out.println(DoubleInfo.INSTANCE.unparse(h));
  }
}