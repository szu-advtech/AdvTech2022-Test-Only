package net.floodlightcontroller.packetstreamer;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.server.TServer;
import org.apache.thrift.server.THsHaServer;
import org.apache.thrift.transport.TFramedTransport;
import org.apache.thrift.transport.TNonblockingServerSocket;
import org.apache.thrift.transport.TNonblockingServerTransport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class PacketStreamerServer {
    protected static Logger log = LoggerFactory.getLogger(PacketStreamerServer.class);
    protected static int port = 9090;
    protected static PacketStreamerHandler handler;
    protected static PacketStreamer.Processor<PacketStreamerHandler> processor;
    public static void main(String [] args) {
        try {
            port = Integer.parseInt(System.getProperty("net.floodlightcontroller.packetstreamer.port", "9090"));
            handler = new PacketStreamerHandler();
            processor = new PacketStreamer.Processor<PacketStreamerHandler>(handler);
            Runnable simple = new Runnable() {
                public void run() {
                    hshaServer(processor);
                }
            };
            new Thread(simple).start();
        } catch (Exception x) {
            x.printStackTrace();
        }
    }
    public static void hshaServer(PacketStreamer.Processor<PacketStreamerHandler> processor) {
        try {
            TNonblockingServerTransport serverTransport = new TNonblockingServerSocket(port);
            THsHaServer.Args args = new THsHaServer.Args(serverTransport);
            args.processor(processor);
            args.transportFactory(new TFramedTransport.Factory());
            args.protocolFactory(new TBinaryProtocol.Factory(true, true));
            TServer server = new THsHaServer(args);
            log.info("Starting the packetstreamer hsha server on port {} ...", port);
            server.serve();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
