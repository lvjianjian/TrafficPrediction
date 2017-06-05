
import java.awt.Graphics2D;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

/**

 * Class DrawGraphics.java 

 * Description  java2D绘制直线，矩形，椭圆，旋转图形

 * Company mapbar  

 * author Chenll 

 * Version 1.0 

 * Date 2012-7-20 下午12:06:15

 */
public class DrawGraphics{

    private BufferedImage image;

    private  Graphics2D graphics;

    public void init(){
        int width=480,hight=720;
        image = new BufferedImage(width,hight,BufferedImage.TYPE_INT_RGB);
        //获取图形上下文
        graphics = (Graphics2D)image.getGraphics();
    }


    /**
     * 创建一个(x1,y1)到(x2,y2)的Line2D对象
     * @throws IOException
     */
    public void drawLine() throws IOException{
        init();
        Line2D line=new Line2D.Double(2,2,300,300);
        graphics.draw(line);
        graphics.dispose();
        outImage("PNG","Line.PNG");
    }


    /**
     * 创建一个左上角坐标是(50,50)，宽是300，高是400的一个矩形对象
     * @throws IOException
     */
    public void drawRect() throws IOException{
        init();
        Rectangle2D rect = new Rectangle2D.Double(50,50,400,400);
        graphics.draw(rect);
        graphics.fill(rect);
        graphics.dispose();
        outImage("PNG","Rect.PNG");
    }

    /**
     * 创建了一个左上角坐标是(50,50)，宽是300，高是200的一个椭圆对象,如果高，宽一样，则是一个标准的圆
     *
     * @throws IOException
     */
    public void drawEllipse() throws IOException{
        init();
        Ellipse2D ellipse=new Ellipse2D.Double(50,50,300,200);
        graphics.draw(ellipse);
        graphics.fill(ellipse);
        graphics.dispose();
        outImage("PNG","ellipse.PNG");
    }

    /**
     * 输出绘制的图形
     * @param type
     * @param filePath
     * @throws IOException
     */
    public void outImage(String type,String filePath) throws IOException{
        ImageIO.write(image,type, new File(filePath));
    }

    public static void main(String[] args) throws IOException{
        DrawGraphics dg = new DrawGraphics();
        dg.drawLine();
        dg.drawRect();
        dg.drawEllipse();
    }
}